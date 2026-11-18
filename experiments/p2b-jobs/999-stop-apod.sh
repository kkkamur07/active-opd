# USER 2026-11-16: pause the GPUs when everything is done. Last job on apod.
# If any job failed in this queue since the stop job was queued, give 2 h for a fix
# (touch ~/queue/KEEP_UP to cancel the stop), then stop this VM anyway.
B=6
if [ "$(grep -c FAILED ~/queue/queue.log)" -gt "$B" ]; then
  echo "$(date -u +%FT%TZ) failures since stop was queued: waiting 2 h before stopping" >> ~/queue/queue.log
  sleep 7200
fi
[ -f ~/queue/KEEP_UP ] && { echo "$(date -u +%FT%TZ) KEEP_UP set: not stopping" >> ~/queue/queue.log; exit 0; }
echo "$(date -u +%FT%TZ) stopping apod" >> ~/queue/queue.log
sudo shutdown -h +1
