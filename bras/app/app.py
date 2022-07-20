import logging
from fastapi import FastAPI
from bras.app.routers import views

logging.basicConfig(level=logging.INFO)

app = FastAPI()


app.include_router(views.router, prefix="/views")
@app.get("/")
async def test():
    return {"status": "ok"}