#!/bin/bash

# IIMS AI Application Startup Script
# This script handles the startup of the IIMS AI application using Docker Compose

set -e  # Exit on any error

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

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if docker-compose is available
check_docker_compose() {
    if ! command -v docker-compose > /dev/null 2>&1 && ! docker compose version > /dev/null 2>&1; then
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    print_success "Docker Compose is available"
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data/datasets
    mkdir -p artifacts/models
    mkdir -p logs
    print_success "Directories created"
}

# Function to set proper permissions
set_permissions() {
    print_status "Setting proper permissions..."
    chmod +x start.sh
    # Ensure data and artifacts directories are writable
    chmod -R 755 data artifacts 2>/dev/null || true
    print_success "Permissions set"
}

# Function to build and start the application
start_application() {
    print_status "Building and starting IIMS AI application..."
    
    # Use docker compose if available, fallback to docker-compose
    if docker compose version > /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    # Build and start services
    $COMPOSE_CMD down --remove-orphans
    $COMPOSE_CMD build --no-cache
    $COMPOSE_CMD up -d
    
    print_success "Application started successfully"
}

# Function to show application status
show_status() {
    print_status "Checking application status..."
    
    # Use docker compose if available, fallback to docker-compose
    if docker compose version > /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    $COMPOSE_CMD ps
    
    # Wait a moment for services to fully start
    sleep 5
    
    # Check if the application is responding
    if curl -f http://localhost:8080/health > /dev/null 2>&1; then
        print_success "Application is healthy and responding at http://localhost:8080"
    else
        print_warning "Application may still be starting up. Check logs with: docker-compose logs -f"
    fi
}

# Function to show logs
show_logs() {
    print_status "Showing application logs..."
    
    # Use docker compose if available, fallback to docker-compose
    if docker compose version > /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    $COMPOSE_CMD logs -f
}

# Function to stop the application
stop_application() {
    print_status "Stopping IIMS AI application..."
    
    # Use docker compose if available, fallback to docker-compose
    if docker compose version > /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    $COMPOSE_CMD down
    print_success "Application stopped"
}

# Function to show help
show_help() {
    echo "IIMS AI Application Startup Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  start     Build and start the application (default)"
    echo "  stop      Stop the application"
    echo "  restart   Restart the application"
    echo "  logs      Show application logs"
    echo "  status    Show application status"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Start the application"
    echo "  $0 start        # Start the application"
    echo "  $0 logs         # Show logs"
    echo "  $0 stop         # Stop the application"
}

# Main execution
main() {
    local action=${1:-start}
    
    case $action in
        start)
            print_status "Starting IIMS AI Application..."
            check_docker
            check_docker_compose
            create_directories
            set_permissions
            start_application
            show_status
            ;;
        stop)
            check_docker
            check_docker_compose
            stop_application
            ;;
        restart)
            print_status "Restarting IIMS AI Application..."
            check_docker
            check_docker_compose
            stop_application
            sleep 2
            start_application
            show_status
            ;;
        logs)
            check_docker
            check_docker_compose
            show_logs
            ;;
        status)
            check_docker
            check_docker_compose
            show_status
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $action"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
