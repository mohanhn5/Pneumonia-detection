# 1. Use a lightweight Python base image
FROM python:3.9-slim

# 2. Set the working directory inside the container to /app
WORKDIR /app

# 3. Copy the requirements file first (for better caching)
COPY requirements.txt .

# 4. Install dependencies
# (We add --no-cache-dir to keep the image size smaller)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code (app.py and model)
COPY . .

# 6. Expose Port 5000 (this is the port Flask uses)
EXPOSE 5000

# 7. Define the command to run your app
CMD ["python", "app.py"]