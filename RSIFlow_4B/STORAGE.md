# Experiment storage

Persistent RSIFlow data belongs under `/root/data/RSI_iclr2027`:

- Code: `rsiH/RSIFlow_4B`.
- Results, trajectories, checkpoints, Task/Meta snapshots: `rsiH/Rollout_logs`.
- Ordinary temporary files: `.runtime/tmp`.
- Experiment caches: `.cache/rsiflow` (configured by `storage_env.sh`).
- Optional remote-worker private state: `.state/RSIFlow_4B`.

The local Meta chroot is a necessary exception: `/tmp/rsiflow_meta_local_<identity>`.
The data volume's Lustre filesystem rejects accesses by its isolated UID with
`Operation not supported`. Do not relocate a live chroot there or loosen `/root`
permissions. The controller stages bounded inputs into the chroot and copies
results back to the run directory; cleanup removes per-call mutable contents.
Relay sockets must be on the same filesystem as this chroot because they are
hard-linked, so their temporary directories explicitly use the runtime parent.
Task evaluation sandboxes also use dedicated `rsi-candidate-*` and
`rsi-environment-*` directories in `/tmp` because they run as isolated UIDs.
They are cleaned up when their context/session closes. General controller
temporary files continue to use the data disk.

The launcher checks total system-filesystem usage plus 4,000,000,000 bytes of
headroom against a ceiling of 20,000,000,000 bytes (decimal GB). Meta checks the
same headroom before each call and checks current usage while the child executes.
These are admission/runtime checks, not a filesystem quota on unrelated programs;
they cannot prevent another process from filling the disk between checks. They
never delete results, model weights, or other users' files.

Storage relocation does not change task manifests, model parameters, generation
settings, scoring, candidate acceptance, or evidence selection. Old failed runs
retain their original protocol fingerprints; continuation with changed source
must be explicit rather than silently editing their protocol files.

Inspect usage with:

```bash
df -h / /root/data
du -sh /tmp/rsiflow_meta_local_* 2>/dev/null
```
