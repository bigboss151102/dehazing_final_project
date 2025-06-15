# Sử dụng CUDA 12.1.1 cho x86_64
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Cập nhật hệ thống & cài đặt các gói cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Cập nhật NVIDIA GPG Key cho x86_64
RUN wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/7fa2af80.pub | tee /usr/share/keyrings/nvidia-keyring.gpg > /dev/null

# Cài đặt Python và các công cụ cần thiết
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Cập nhật pip và cài đặt PyTorch với CUDA 12.1 (phiên bản phù hợp cho x86_64)
RUN pip3 install --upgrade pip \
    && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Đặt thư mục làm việc
WORKDIR /app

# Sao chép mã nguồn vào container
COPY . /app/

# Cấp quyền chạy cho script huấn luyện
RUN chmod +x trainDENSE.sh

# Cài đặt các thư viện Python từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Cấu hình môi trường CUDA
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=1

# Kiểm tra CUDA có hoạt động không
RUN python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# Chạy huấn luyện mô hình
CMD ["bash", "trainDENSE.sh"]
