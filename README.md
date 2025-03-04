# Setup

```
# Setup
python3 -m venv env
source env/bin/activate
pip install ultralytics
pip install "fastapi[standard]"

# Run
fastapi run main.py --host 0.0.0.0 --port 80
```