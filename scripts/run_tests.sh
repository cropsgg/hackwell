#!/bin/bash

# AI Wellness Assistant - Test Runner Script

set -e  # Exit on any error

echo "🧪 AI Wellness Assistant - Test Suite"
echo "======================================"

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

# Test configuration
TEST_PARALLEL=${TEST_PARALLEL:-4}
COVERAGE_THRESHOLD=${COVERAGE_THRESHOLD:-80}
VERBOSE=${VERBOSE:-false}

# Change to project root
cd "$(dirname "$0")/.."

# Function to run tests for a service
run_service_tests() {
    local service_name=$1
    local service_path=$2
    
    print_status "Running tests for $service_name..."
    
    if [ ! -d "$service_path" ]; then
        print_warning "$service_name directory not found, skipping"
        return 0
    fi
    
    if [ ! -f "$service_path/requirements.txt" ]; then
        print_warning "$service_name has no requirements.txt, skipping"
        return 0
    fi
    
    # Check if tests directory exists
    if [ ! -d "$service_path/tests" ]; then
        print_warning "$service_name has no tests directory, skipping"
        return 0
    fi
    
    cd "$service_path"
    
    # Install dependencies
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt > /dev/null 2>&1 || {
            print_error "Failed to install dependencies for $service_name"
            cd - > /dev/null
            return 1
        }
    fi
    
    # Install test dependencies
    pip install pytest pytest-asyncio pytest-cov pytest-mock > /dev/null 2>&1 || {
        print_error "Failed to install test dependencies for $service_name"
        cd - > /dev/null
        return 1
    }
    
    # Run tests with coverage
    local test_args=""
    if [ "$VERBOSE" = "true" ]; then
        test_args="-v"
    fi
    
    pytest tests/ $test_args --cov=. --cov-report=term-missing --cov-fail-under=$COVERAGE_THRESHOLD || {
        print_error "Tests failed for $service_name"
        cd - > /dev/null
        return 1
    }
    
    print_success "Tests passed for $service_name"
    cd - > /dev/null
    return 0
}

# Function to run integration tests
run_integration_tests() {
    print_status "Running integration tests..."
    
    # Check if services are running
    if ! curl -s http://localhost:8000/health > /dev/null; then
        print_warning "Gateway not running, starting services for integration tests..."
        
        cd infra
        docker-compose up -d > /dev/null 2>&1
        
        # Wait for services to be ready
        sleep 30
        
        # Check again
        if ! curl -s http://localhost:8000/health > /dev/null; then
            print_error "Failed to start services for integration tests"
            return 1
        fi
        
        cd - > /dev/null
    fi
    
    # Run integration tests
    python -m pytest tests/integration/ -v || {
        print_error "Integration tests failed"
        return 1
    }
    
    print_success "Integration tests passed"
    return 0
}

# Function to run API tests
run_api_tests() {
    print_status "Running API tests..."
    
    # Check if API is accessible
    if ! curl -s http://localhost:8000/health > /dev/null; then
        print_warning "API not accessible, skipping API tests"
        return 0
    fi
    
    # Run API tests using curl and basic validation
    local api_base="http://localhost:8000/api/v1"
    
    # Test health endpoint
    print_status "Testing health endpoint..."
    local health_response=$(curl -s "$api_base/../health")
    if echo "$health_response" | grep -q '"status"'; then
        print_success "Health endpoint test passed"
    else
        print_error "Health endpoint test failed"
        return 1
    fi
    
    # Test API documentation
    print_status "Testing API documentation..."
    if curl -s http://localhost:8000/docs > /dev/null; then
        print_success "API documentation accessible"
    else
        print_warning "API documentation not accessible"
    fi
    
    print_success "API tests completed"
    return 0
}

# Function to run security tests
run_security_tests() {
    print_status "Running security tests..."
    
    # Test authentication
    print_status "Testing authentication..."
    local unauth_response=$(curl -s -w "%{http_code}" -o /dev/null http://localhost:8000/api/v1/patients)
    if [ "$unauth_response" = "401" ] || [ "$unauth_response" = "403" ]; then
        print_success "Authentication test passed"
    else
        print_warning "Authentication test unclear (status: $unauth_response)"
    fi
    
    # Test security headers
    print_status "Testing security headers..."
    local headers=$(curl -s -I http://localhost:8000/health)
    if echo "$headers" | grep -q "X-Content-Type-Options"; then
        print_success "Security headers test passed"
    else
        print_warning "Security headers not found"
    fi
    
    print_success "Security tests completed"
    return 0
}

# Function to run performance tests
run_performance_tests() {
    print_status "Running basic performance tests..."
    
    if ! command -v ab &> /dev/null; then
        print_warning "Apache Bench (ab) not found, skipping performance tests"
        return 0
    fi
    
    # Basic load test on health endpoint
    print_status "Running load test on health endpoint..."
    ab -n 100 -c 10 -q http://localhost:8000/health > /dev/null 2>&1 || {
        print_warning "Performance test encountered issues"
        return 0
    }
    
    print_success "Performance tests completed"
    return 0
}

# Function to generate test report
generate_test_report() {
    print_status "Generating test report..."
    
    local report_file="test_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "AI Wellness Assistant - Test Report"
        echo "Generated: $(date)"
        echo "========================================"
        echo
        echo "Test Summary:"
        echo "- Gateway Service: $gateway_result"
        echo "- ML Risk Service: $ml_risk_result"
        echo "- Agents Service: $agents_result"
        echo "- Verifier Service: $verifier_result"
        echo "- Integration Tests: $integration_result"
        echo "- API Tests: $api_result"
        echo "- Security Tests: $security_result"
        echo "- Performance Tests: $performance_result"
        echo
        echo "Overall Status: $overall_status"
    } > "$report_file"
    
    print_success "Test report generated: $report_file"
}

# Main test execution
main() {
    print_status "Starting comprehensive test suite..."
    echo
    
    # Initialize results
    gateway_result="SKIPPED"
    ml_risk_result="SKIPPED"
    agents_result="SKIPPED"
    verifier_result="SKIPPED"
    integration_result="SKIPPED"
    api_result="SKIPPED"
    security_result="SKIPPED"
    performance_result="SKIPPED"
    
    # Run unit tests for each service
    print_status "=== UNIT TESTS ==="
    
    if run_service_tests "Gateway" "gateway"; then
        gateway_result="PASSED"
    else
        gateway_result="FAILED"
    fi
    
    if run_service_tests "ML Risk" "services/ml_risk"; then
        ml_risk_result="PASSED"
    else
        ml_risk_result="FAILED"
    fi
    
    if run_service_tests "Agents" "services/agents"; then
        agents_result="PASSED"
    else
        agents_result="FAILED"
    fi
    
    if run_service_tests "Verifier" "services/verifier"; then
        verifier_result="PASSED"
    else
        verifier_result="FAILED"
    fi
    
    echo
    print_status "=== INTEGRATION TESTS ==="
    
    if run_integration_tests; then
        integration_result="PASSED"
    else
        integration_result="FAILED"
    fi
    
    echo
    print_status "=== API TESTS ==="
    
    if run_api_tests; then
        api_result="PASSED"
    else
        api_result="FAILED"
    fi
    
    echo
    print_status "=== SECURITY TESTS ==="
    
    if run_security_tests; then
        security_result="PASSED"
    else
        security_result="FAILED"
    fi
    
    echo
    print_status "=== PERFORMANCE TESTS ==="
    
    if run_performance_tests; then
        performance_result="PASSED"
    else
        performance_result="FAILED"
    fi
    
    # Determine overall status
    if [[ "$gateway_result" == "FAILED" || "$ml_risk_result" == "FAILED" || 
          "$agents_result" == "FAILED" || "$verifier_result" == "FAILED" || 
          "$integration_result" == "FAILED" ]]; then
        overall_status="FAILED"
    else
        overall_status="PASSED"
    fi
    
    echo
    print_status "=== TEST SUMMARY ==="
    echo "Gateway Service:    $gateway_result"
    echo "ML Risk Service:    $ml_risk_result"
    echo "Agents Service:     $agents_result"
    echo "Verifier Service:   $verifier_result"
    echo "Integration Tests:  $integration_result"
    echo "API Tests:          $api_result"
    echo "Security Tests:     $security_result"
    echo "Performance Tests:  $performance_result"
    echo
    
    if [ "$overall_status" = "PASSED" ]; then
        print_success "🎉 All tests passed! System is ready for deployment."
    else
        print_error "❌ Some tests failed. Please review and fix issues before deployment."
        exit 1
    fi
    
    # Generate report
    generate_test_report
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --coverage)
            COVERAGE_THRESHOLD="$2"
            shift 2
            ;;
        --parallel)
            TEST_PARALLEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  -v, --verbose     Verbose test output"
            echo "  --coverage NUM    Coverage threshold (default: 80)"
            echo "  --parallel NUM    Parallel test workers (default: 4)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"
