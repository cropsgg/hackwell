#!/bin/bash

# AI Wellness Assistant - Local Development Bootstrap Script

set -e  # Exit on any error

echo "🏥 AI Wellness Assistant - Local Development Setup"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    print_status "Checking Docker availability..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    print_success "Docker is available and running"
}

# Create environment file
create_env_file() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        print_status "Creating .env file from template..."
        cp env.example .env
        
        # Generate a secret key
        SECRET_KEY=$(openssl rand -hex 32 2>/dev/null || echo "your-secret-key-change-in-production")
        
        # Update .env with local values
        sed -i.bak "s|your-secret-key-change-in-production|$SECRET_KEY|g" .env
        sed -i.bak "s|https://your-project.supabase.co|http://localhost:54321|g" .env
        sed -i.bak "s|your-anon-key|demo-anon-key|g" .env
        sed -i.bak "s|your-service-key|demo-service-key|g" .env
        
        rm .env.bak 2>/dev/null || true
        
        print_success "Environment file created"
        print_warning "Please update .env with your actual API keys for external services"
    else
        print_success "Environment file already exists"
    fi
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    cd infra
    docker-compose build --no-cache
    
    print_success "Docker images built successfully"
}

# Start services
start_services() {
    print_status "Starting services with Docker Compose..."
    
    cd infra
    docker-compose up -d
    
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    check_service_health
}

# Check service health
check_service_health() {
    print_status "Checking service health..."
    
    # Check database
    if docker-compose exec -T postgres pg_isready -U postgres -d wellness; then
        print_success "Database is ready"
    else
        print_warning "Database may still be starting up"
    fi
    
    # Check gateway
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "Gateway is ready"
    else
        print_warning "Gateway may still be starting up"
    fi
}

# Initialize database
init_database() {
    print_status "Initializing database schema and seed data..."
    
    cd infra
    
    # Wait for database to be ready
    print_status "Waiting for database to be fully ready..."
    sleep 10
    
    # Check if schema is already applied
    if docker-compose exec -T postgres psql -U postgres -d wellness -c "SELECT COUNT(*) FROM patients;" 2>/dev/null; then
        print_success "Database already initialized"
        return
    fi
    
    # Apply schema and seed data (these are applied automatically via docker-entrypoint-initdb.d)
    print_success "Database initialization completed"
}

# Generate synthetic data
generate_synthetic_data() {
    print_status "Generating additional synthetic data..."
    
    # This would run the synthetic data generation script
    # For now, we use the seed data already in the database
    print_success "Using seed data from database initialization"
}

# Train ML models
train_models() {
    print_status "Training ML models..."
    
    cd infra
    
    # Check if ML service is ready
    if ! docker-compose ps ml_risk | grep -q "Up"; then
        print_warning "ML Risk service not running, skipping model training"
        return
    fi
    
    # Generate synthetic training data and train model
    docker-compose exec ml_risk python train.py \
        --generate-data \
        --n-samples 2000 \
        --algo lightgbm \
        --output models/risk_lgbm_v0_1.bin \
        --cv-folds 3
    
    print_success "ML model training completed"
}

# Run tests
run_tests() {
    print_status "Running basic health checks..."
    
    cd infra
    
    # Test gateway
    echo "Testing Gateway..."
    curl -s http://localhost:8000/health | jq '.' || print_warning "Gateway health check failed"
    
    # Test ML service
    echo "Testing ML Risk Service..."
    curl -s http://localhost:8001/health | jq '.' || print_warning "ML Risk service health check failed"
    
    # Test database connection
    echo "Testing Database..."
    docker-compose exec -T postgres psql -U postgres -d wellness -c "SELECT 'Database OK' as status;" || print_warning "Database test failed"
    
    print_success "Basic health checks completed"
}

# Show service URLs
show_urls() {
    echo ""
    echo "🚀 AI Wellness Assistant is now running!"
    echo "========================================"
    echo ""
    echo "Service URLs:"
    echo "• API Gateway:      http://localhost:8000"
    echo "• API Documentation: http://localhost:8000/docs"
    echo "• pgAdmin:          http://localhost:8080 (admin@wellness.com / admin)"
    echo "• ML Risk Service:  http://localhost:8001"
    echo "• Agents Service:   http://localhost:8002"
    echo "• Verifier Service: http://localhost:8003"
    echo "• Ingest Service:   http://localhost:8004"
    echo "• WebSocket Stream: http://localhost:8005"
    echo ""
    echo "Database:"
    echo "• Host: localhost:5432"
    echo "• Database: wellness"
    echo "• Username: postgres"
    echo "• Password: password"
    echo ""
    echo "Demo Patient ID: f47ac10b-58cc-4372-a567-0e02b2c3d479 (Maria Gonzalez)"
    echo "Demo Clinician ID: c47ac10b-58cc-4372-a567-0e02b2c3d480 (Dr. Sarah Patel)"
    echo ""
    echo "Useful commands:"
    echo "• View logs: cd infra && docker-compose logs -f"
    echo "• Stop services: cd infra && docker-compose down"
    echo "• Reset database: cd infra && make db_reset"
    echo "• Run demo: cd infra && make demo"
    echo ""
}

# Main execution
main() {
    echo "Starting local development setup..."
    echo ""
    
    # Change to project root
    cd "$(dirname "$0")/.."
    
    # Run setup steps
    check_docker
    create_env_file
    build_images
    start_services
    init_database
    generate_synthetic_data
    train_models
    run_tests
    
    show_urls
    
    print_success "Setup completed successfully! 🎉"
    echo ""
    echo "Next steps:"
    echo "1. Review the .env file and update API keys"
    echo "2. Visit http://localhost:8000/docs to explore the API"
    echo "3. Check service logs if needed: cd infra && docker-compose logs -f"
    echo ""
}

# Run main function
main "$@"
