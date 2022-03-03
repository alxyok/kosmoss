#!/usr/bin/bash 

export MAX_WORKERS=$(python -c "import psutil; print(psutil.cpu_count(logical=False))")

# Usually, you should enable pylint, really
# But because PyTorch generates errors on its own, we'll simplify by just disabling it
# Our code is clean though ;)
USERNAME='mluser' python flows.py --no-pylint \
    run \
        --max-num-splits 7000 \
        --max-workers ${MAX_WORKERS} >> ${HOME}/.kosmoss/logs/build_graphs.stdout