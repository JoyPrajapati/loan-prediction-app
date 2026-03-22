# Step 1 — Start from an official Python base image
# Think of this as: "start with a clean PC that has Python 3.9 installed"
FROM python:3.9-slim

# Step 2 — Set the working directory inside the container
# All commands after this run inside /app
WORKDIR /app

# Step 3 — Copy requirements first (smart caching trick)
# Docker caches layers — if requirements didn't change, it skips reinstalling
COPY requirements.txt .

# Step 4 — Install all Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Step 5 — Copy everything else into the container
COPY . .

# Step 6 — Tell Docker your app runs on port 5000
EXPOSE 5000

# Step 7 — The command that starts your app when container runs
CMD ["python", "app.py"]


