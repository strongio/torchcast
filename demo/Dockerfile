FROM python:3.9

WORKDIR /app

RUN python3 -m pip install --upgrade pip setuptools && \
    python3 -m pip install gunicorn

COPY setup.py .

# Install only the dependencies
RUN python setup.py egg_info && \
    pip install -r *.egg-info/requires.txt && \
    rm -rf *.egg-info/

COPY  . .

RUN pip install .[demo]

EXPOSE 8097

CMD ["gunicorn", "--bind", "0.0.0.0:8097", "demo.app:server"]
