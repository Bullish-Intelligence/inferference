# inferference — the llama.cpp router as a systemd --user unit.
#
# Import from your NixOS configuration:
#
#     imports = [ /path/to/inferference/nix/nixos-module.nix ];
#
#     services.inferference-router = {
#       enable  = true;
#       user    = "youruser";
#       repoDir = "/path/to/inferference";
#     };
#
# WHAT THIS MODULE DOES NOT HOLD, on purpose: no model path, no port, no
# adapter pool, no device placement, no context size, no `--models-max`. Those
# all live in YAML under ~/.config/inferference/ — placement.yaml (which models
# are resident, on which device(s), and how the router itself is launched) and
# models/<name>.yaml (what each model is) — read by the unit AT START. Changing
# a model, a pool, or the resident set is an edit and a
# `systemctl --user restart`, never a NixOS rebuild. See
# ci/runner/placement.example.yaml and ci/runner/model.example.yaml.
#
# ONE ROUTER, TOTAL. This replaces the retired single-role design —
# services.inferference-server, one instance per role, one model pinned per
# card via $LLAMA_GPU_<ROLE>. AGENTS.md now says "THIS LIBRARY OWNS ALL LLM
# SERVING. There is no other server.": every model this machine serves is a
# child process the ONE router spawns from a generated preset INI, and which
# card(s) a model lands on is a placement.yaml decision, never a unit
# parameter and never a literal device index anywhere in Nix.
#
# The reconciler (ci/runner/inferference-reconcile.py) is the operational tool
# for changing what is resident without restarting the router:
#   inferference-reconcile.py plan     # what would change, and the VRAM budget
#   inferference-reconcile.py apply    # do it
#   inferference-reconcile.py status   # what is resident now, and its health
#   inferference-reconcile.py evict <device>   # free a card; config untouched
# None of that is Nix's job — a reconcile is a config edit plus a CLI call, not
# a rebuild.
{ config, lib, pkgs, ... }:

let
  cfg = config.services.inferference-router;

  # The launcher renders the preset INI from placement.yaml + models/*.yaml,
  # resolves the Mode A artefact, and asserts the loader resolves libcuda to
  # the real driver rather than the nix CUDA stub before it starts. Every
  # missing fact exits 78 (EX_CONFIG), which systemd reports distinctly, so a
  # misconfiguration never looks like a crashed router.
  launcher = "${cfg.repoDir}/ci/runner/inferference-router-launch.sh";
in
{
  options.services.inferference-router = {
    enable = lib.mkEnableOption "the inferference llama.cpp router user unit";

    user = lib.mkOption {
      type = lib.types.str;
      description = "User whose systemd --user manager runs the router.";
    };

    repoDir = lib.mkOption {
      type = lib.types.str;
      example = "/path/to/inferference";
      description = ''
        Absolute path to the inferference checkout. The unit runs
        ci/runner/inferference-router-launch.sh from here, which reads the
        published Mode A artefact at ci/modea/server/current.

        This is the one project fact the module has to learn. It is an option
        rather than a hardcoded path so the module itself stays generic.
      '';
    };

    configDir = lib.mkOption {
      type = lib.types.str;
      default = "%h/.config/inferference";
      description = ''
        Where placement.yaml and models/*.yaml live. %h expands to the user's
        home. This is WHERE the config is, never WHAT is in it — host, port,
        `--models-max`, the resident set and every model's device placement
        are all declared inside that directory, not here.
      '';
    };

    autoStart = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Start the router at login / on boot with lingering enabled.";
    };

    lingering = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Enable systemd lingering for the user, so the router survives logout
        and starts at boot. Without it an "always-on" router is only on while
        you are logged in.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    users.users.${cfg.user}.linger = lib.mkIf cfg.lingering true;

    systemd.user.services.inferference-router = {
      description = "inferference — llama.cpp router (all resident models)";
      documentation = [ "file://${cfg.repoDir}/AGENTS.md" ];
      wantedBy = lib.optional cfg.autoStart "default.target";

      # No device placement and no model path here. The launcher renders the
      # preset INI and resolves everything else from
      # ${cfg.configDir}/{placement.yaml,models/*.yaml}.
      environment.INFERFERENCE_CONFIG_DIR = cfg.configDir;

      serviceConfig = {
        Type = "exec";
        # --check does the whole resolution (artefact, loader, config, preset
        # render) without starting the router, so a bad config fails in
        # ExecStartPre with EX_CONFIG rather than as a crash loop.
        ExecStartPre = "${launcher} --check";
        ExecStart = launcher;
        Restart = "on-failure";
        # Guard thermal stops and failed safety admission need a cool/manual restart.
        RestartPreventExitStatus = "75 76";
        RestartSec = 5;
        # Startup loads every resident model (load-on-startup=true in the
        # rendered preset); a wide resident set is slow. Do not let systemd
        # kill a healthy start.
        TimeoutStartSec = 300;
        # A stop must reach the llama-server processes EXACTLY ONCE. With the
        # default KillMode=control-group, systemd broadcasts KillSignal to the
        # whole cgroup; the guard (MainPID) dies at once because it traps
        # nothing, and systemd then re-broadcasts KillSignal in final-sigterm.
        # Two SIGINTs about 1.5 ms apart make llama-server call exit() from
        # inside its signal handler, concurrently with its own shutdown. That
        # deadlocked 4 of 38 stops.
        # KillMode=mixed signals only this unit's main process, and
        # gpu-thermal-guard.sh relays one signal into the process group it owns
        # and waits for it. The guard's relay is required: mixed without it
        # leaves the router unsignalled and SIGKILLed.
        KillMode = "mixed";
        # The longest clean stop measured is 13.54 s. 90 s is headroom for a
        # wide resident set, not a workaround for the defect above.
        TimeoutStopSec = 90;
        KillSignal = "SIGINT";
        NoNewPrivileges = true;
        # The launcher reads the artefact, the models and the YAML out of $HOME.
        ProtectHome = false;
        PrivateTmp = false;
      };
    };
  };
}
