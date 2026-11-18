# USER 2026-11-16: pause the GPUs when everything is done. Last job on apod2.
# Wait (<= 6 h) for the laptop relay to copy the seed-3 results (no checkpoints) to apod and
# write ~/COPIED; then stop this VM (guest shutdown = instance stopped; boot disk kept).
for i in $(seq 1 72); do [ -f ~/COPIED ] && break; sleep 300; done
echo "$(date -u +%FT%TZ) stopping apod2 (copied: $([ -f ~/COPIED ] && echo yes || echo NO))" >> ~/queue/queue.log
sudo shutdown -h +1
