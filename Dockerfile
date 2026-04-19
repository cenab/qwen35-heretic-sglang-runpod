FROM lmsysorg/sglang:dev

RUN python3 -m pip install --no-cache-dir runpod requests hf_transfer

WORKDIR /workspace
COPY handler.py /workspace/handler.py

CMD ["python3", "/workspace/handler.py"]
