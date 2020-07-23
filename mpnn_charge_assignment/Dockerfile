FROM python:3.7
WORKDIR /app

# COPY ./requirements.txt .
# RUN pip install -r requirements.txt

RUN pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch-scatter==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip install torch-sparse==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip install torch-cluster==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip install torch-spline-conv==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.5.0.html
RUN pip install torch-geometric
COPY . .
# CMD python deployment_main.py "temp" "KAXQIL_clean.cif"