FROM python:3.10-slim

# Create working directory
WORKDIR /app

# Copy requirements first (for better cache)
COPY requirements.txt .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the project (including savedmodel.pth & app.py)
COPY . .

# Expose Flask port
EXPOSE 5001

# Run the Flask app
CMD ["python", "app.py"]