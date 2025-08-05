# rag-projects

## Install 
Start your virtual env
```
uv venv && source .venv/bin/activate
```
Install 
```
uv sync
```
## How to run
Go to each folder, start streamlit
```
streamlit run --server.port 8501 app.py
```

or start fastapi server
```
uvicorn main:app --host 0.0.0.0 --port 8081 --reload
```

## Reference
