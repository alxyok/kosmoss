#!/usr/bin/bash 

export MAX_WORKERS=$(python -c "import psutil; print(psutil.cpu_count(logical=False))")

USERNAME='mluser' python flows.py \
    run \
        --max-num-splits 7000 \
        --max-workers ${MAX_WORKERS} >> ${HOME}/.kosmoss/logs/build_graphs.stdout