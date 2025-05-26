python -m lerobot.inav.parse_dataset \
    --csv-dir /home/demo/lerobot-beavr/lerobot/inav/datasets/moon_lander_raw/states \
    --image-dir /home/demo/lerobot-beavr/lerobot/inav/datasets/moon_lander_raw/descent_imgs \
    --output-dir /home/demo/lerobot-beavr/lerobot/inav/datasets/moon_lander_lerobot \
    --repo-id aposadasn/moon_lander \
    --image-height=256 \
    --image-width=256 \
    --push
