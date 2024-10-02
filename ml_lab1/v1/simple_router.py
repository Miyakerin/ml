from fastapi import APIRouter, UploadFile, File
import uuid

from src.core.models.classificator import Classificator

simple_router = APIRouter()

@simple_router.post('/wav_file')
def post_wav_file(file: UploadFile):
    file_uuid = str(uuid.uuid4())
    with open(('static/' + file_uuid), 'wb') as out_file:
        out_file.write(file.file.read())
    return {'message': f'file uploaded successfully with uuid: {file_uuid}'}


@simple_router.post('/wav_file/{uuid}/analyze')
def classificate_wav_file(uuid: str) -> dict[str, str]:
    classificator = Classificator()
    result = classificator.classificate(uuid)
    return {"emotion": result}

@simple_router.post('/wav_file/analyze')
def load_wav_and_classificate(file: UploadFile = File(...)):
    file_uuid = str(uuid.uuid4())
    with open(('static/' + file_uuid), 'wb') as out_file:
        out_file.write(file.file.read())
    classificator = Classificator()
    result = classificator.classificate(file_uuid)
    print(result)
    return {"emotion": result}
