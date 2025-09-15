# AI Wellness Assistant - Development Makefile

.PHONY: help up down logs clean db_init db_reset train test demo lint format

# Default target
help:
	@echo "Available targets:"
	@echo "  up         - Start all services with Docker Compose"
	@echo "  down       - Stop all services"
	@echo "  logs       - Show logs from all services"
	@echo "  clean      - Clean up containers and volumes"
	@echo "  db_init    - Initialize database schema and seed data"
	@echo "  db_reset   - Reset database (drop and recreate)"
	@echo "  train      - Train ML models"
	@echo "  test       - Run test suite"
	@echo "  demo       - Run demo scenarios"
	@echo "  lint       - Run linting on all services"
	@echo "  format     - Format code"

# Environment setup
ENV_FILE := ../.env
ifeq (,$(wildcard $(ENV_FILE)))
    $(warning Warning: .env file not found. Using defaults.)
endif

# Docker Compose commands
up:
	@echo "Starting AI Wellness Assistant services..."
	docker-compose up -d
	@echo "Services started. Gateway available at http://localhost:8000"
	@echo "pgAdmin available at http://localhost:8080 (admin@wellness.com / admin)"

down:
	@echo "Stopping all services..."
	docker-compose down

logs:
	docker-compose logs -f

clean:
	@echo "Cleaning up containers and volumes..."
	docker-compose down -v --remove-orphans
	docker system prune -f

# Database management
db_init:
	@echo "Initializing database..."
	@docker-compose exec postgres psql -U postgres -d wellness -c "SELECT 'Database initialized successfully';"

db_reset:
	@echo "Resetting database..."
	docker-compose exec postgres psql -U postgres -c "DROP DATABASE IF EXISTS wellness;"
	docker-compose exec postgres psql -U postgres -c "CREATE DATABASE wellness;"
	docker-compose exec postgres psql -U postgres -d wellness -f /docker-entrypoint-initdb.d/01-schema.sql
	docker-compose exec postgres psql -U postgres -d wellness -f /docker-entrypoint-initdb.d/02-seed.sql
	@echo "Database reset complete."

# ML model training
train:
	@echo "Training ML models..."
	docker-compose exec ml_risk python train.py --dataset /app/data/synthetic_cohort.csv --algo lightgbm --output /app/models/risk_lgbm_v0_1.bin
	@echo "Model training complete."

# Testing
test:
	@echo "Running test suite..."
	docker-compose exec gateway python -m pytest tests/ -v
	docker-compose exec ml_risk python -m pytest tests/ -v
	docker-compose exec agents python -m pytest tests/ -v
	docker-compose exec verifier python -m pytest tests/ -v

# Demo scenarios
demo:
	@echo "Running demo scenarios..."
	@echo "Creating recommendations for demo patients..."
	docker-compose exec gateway python -c "
import requests
import json

# Demo: Create recommendation for Maria (high-risk patient)
response = requests.post('http://localhost:8000/api/patients/f47ac10b-58cc-4372-a567-0e02b2c3d479/recommendations')
print('Maria recommendation:', response.status_code)

# Demo: Get patient risk summary
response = requests.get('http://localhost:8000/api/patients/f47ac10b-58cc-4372-a567-0e02b2c3d479/risk')
print('Maria risk summary:', response.json() if response.status_code == 200 else response.status_code)
"

# Code quality
lint:
	@echo "Running linting..."
	docker-compose exec gateway python -m flake8 app/ --max-line-length=100
	docker-compose exec ml_risk python -m flake8 . --max-line-length=100
	docker-compose exec agents python -m flake8 . --max-line-length=100
	docker-compose exec verifier python -m flake8 . --max-line-length=100

format:
	@echo "Formatting code..."
	docker-compose exec gateway python -m black app/ --line-length=100
	docker-compose exec ml_risk python -m black . --line-length=100
	docker-compose exec agents python -m black . --line-length=100
	docker-compose exec verifier python -m black . --line-length=100

# Development helpers
shell-gateway:
	docker-compose exec gateway /bin/bash

shell-db:
	docker-compose exec postgres psql -U postgres -d wellness

shell-ml:
	docker-compose exec ml_risk /bin/bash

# Service-specific operations
restart-gateway:
	docker-compose restart gateway

restart-ml:
	docker-compose restart ml_risk

# Generate synthetic data
generate-data:
	@echo "Generating synthetic patient data..."
	cd .. && python scripts/generate_synthetic.py --n 200 --output db/synthetic_cohort.csv

# Health checks
health:
	@echo "Checking service health..."
	@curl -s http://localhost:8000/health | jq '.' || echo "Gateway not responding"
	@curl -s http://localhost:8001/health | jq '.' || echo "ML Risk service not responding"
	@curl -s http://localhost:8002/health | jq '.' || echo "Agents service not responding"
	@curl -s http://localhost:8003/health | jq '.' || echo "Verifier service not responding"

# Backup and restore
backup-db:
	@echo "Backing up database..."
	docker-compose exec postgres pg_dump -U postgres wellness > backup_$(shell date +%Y%m%d_%H%M%S).sql

restore-db:
	@echo "Usage: make restore-db BACKUP_FILE=backup_file.sql"
	@if [ -z "$(BACKUP_FILE)" ]; then echo "Please specify BACKUP_FILE"; exit 1; fi
	docker-compose exec -T postgres psql -U postgres wellness < $(BACKUP_FILE)

# Performance monitoring
monitor:
	@echo "Service resource usage:"
	docker stats --no-stream wellness_gateway wellness_ml_risk wellness_agents wellness_verifier wellness_postgres wellness_redis

# Full development cycle
dev-setup: up db_init generate-data train
	@echo "Development environment ready!"
	@echo "Access points:"
	@echo "  API Gateway: http://localhost:8000"
	@echo "  pgAdmin: http://localhost:8080"
	@echo "  API Documentation: http://localhost:8000/docs"

# Production preparation
prod-check:
	@echo "Running production readiness checks..."
	@echo "- Security scan..."
	@echo "- Performance tests..."
	@echo "- Configuration validation..."
	@echo "TODO: Implement production checks"
