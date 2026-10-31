uv run scripts/run_agent.py --port 5555
java -Xmx2048m -Dmagicnet.actions=socket:127.0.0.1:5555 \
        -Dmagicnet.jsonl=/tmp/live.jsonl \
        -cp "build:/usr/share/java/forge-gui-desktop.jar" \
        magicnet.bridge.Harness --decks /tmp/pio_41 --games 1 \
        --out /tmp/live --seed 42 --timeout 900