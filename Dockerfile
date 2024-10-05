FROM python:3.11-slim

WORKDIR /DoubleParkingViolation

RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /DoubleParkingViolation

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install "uvicorn[standard]"

COPY . /DoubleParkingViolation

ENV PORT 8080

EXPOSE 8080
HEALTHCHECK CMD curl --fail http://localhost:8080/health || exit 1
# ENV GOOGLE_APPLICATION_CREDENTIALS="capstone-t5-4494681f8d0c.json"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]