#!/bin/bash
# Script to manage persistent LLM models in Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}╔════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  $1${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════╝${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Check if containers are running
check_containers() {
    if ! docker ps | grep -q "open-core-graph-rag-ollama"; then
        print_error "Ollama container is not running. Run './auto launch' first."
        exit 1
    fi
}

# List available models
list_models() {
    print_header "Available Models in Volume"
    
    check_docker
    
    # Check if volume exists
    if ! docker volume ls | grep -q "open-core-graph-rag_ollama-models"; then
        print_warning "No models downloaded yet. Run './auto launch' to download."
        return
    fi
    
    # Try to list from running container
    if docker ps | grep -q "open-core-graph-rag-ollama"; then
        echo "Models currently available:"
        docker exec open-core-graph-rag-ollama ollama list || print_warning "Unable to list models"
    else
        print_warning "Ollama container not running. Models are cached in volume but can't check status."
        echo "Run './auto launch' to start the container and access models."
    fi
}

# Check volume details
volume_info() {
    print_header "LLM Models Volume Information"
    
    check_docker
    
    VOLUME_NAME="open-core-graph-rag_ollama-models"
    
    if ! docker volume ls | grep -q "$VOLUME_NAME"; then
        print_warning "Volume does not exist yet. Run './auto launch' to create it."
        return
    fi
    
    echo "Volume Details:"
    docker volume inspect "$VOLUME_NAME"
    
    echo -e "\n${BLUE}Storage Location:${NC}"
    echo "  On macOS: ~/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw"
    echo "  (Docker Desktop manages the actual path)"
}

# Download a specific model
download_model() {
    print_header "Download LLM Model"
    
    check_docker
    check_containers
    
    if [ -z "$1" ]; then
        print_error "Model name required. Usage: ./manage_models.sh download <model-name>"
        echo ""
        echo "Available models:"
        echo "  - mistral         (5-7GB, recommended)"
        echo "  - neural-chat     (3-5GB)"
        echo "  - llama2          (7GB)"
        echo "  - dolphin-mixtral (45GB)"
        echo "  - zephyr          (3-4GB)"
        exit 1
    fi
    
    MODEL_NAME=$1
    
    print_info "Downloading $MODEL_NAME..."
    print_warning "This may take 5-20 minutes depending on model size and internet speed"
    echo ""
    
    docker exec open-core-graph-rag-ollama ollama pull "$MODEL_NAME"
    
    if [ $? -eq 0 ]; then
        print_success "Successfully downloaded $MODEL_NAME"
        echo ""
        print_info "Model is now cached and will persist across container restarts."
    else
        print_error "Failed to download $MODEL_NAME"
        exit 1
    fi
}

# Remove a model to free space
remove_model() {
    print_header "Remove LLM Model"
    
    check_docker
    check_containers
    
    if [ -z "$1" ]; then
        print_error "Model name required. Usage: ./manage_models.sh remove <model-name>"
        echo ""
        echo "Run './manage_models.sh list' to see available models"
        exit 1
    fi
    
    MODEL_NAME=$1
    
    print_warning "Are you sure you want to remove $MODEL_NAME? (y/n)"
    read -r CONFIRM
    
    if [ "$CONFIRM" != "y" ]; then
        print_info "Cancelled."
        exit 0
    fi
    
    docker exec open-core-graph-rag-ollama ollama rm "$MODEL_NAME"
    
    if [ $? -eq 0 ]; then
        print_success "Successfully removed $MODEL_NAME"
        echo ""
        print_info "Freed up storage space. Model can be re-downloaded with 'download' command."
    else
        print_error "Failed to remove $MODEL_NAME"
        exit 1
    fi
}

# Setup default models
setup_defaults() {
    print_header "Setup Default Models"
    
    check_docker
    check_containers
    
    MODELS=("mistral")  # Add more defaults as needed
    
    for MODEL in "${MODELS[@]}"; do
        if docker exec open-core-graph-rag-ollama ollama list | grep -q "$MODEL"; then
            print_success "$MODEL already downloaded"
        else
            print_info "Downloading $MODEL..."
            docker exec open-core-graph-rag-ollama ollama pull "$MODEL"
        fi
    done
    
    print_success "Default models setup complete!"
}

# Clean up (delete volume)
cleanup() {
    print_header "Cleanup Volume"
    
    check_docker
    
    VOLUME_NAME="open-core-graph-rag_ollama-models"
    
    if ! docker volume ls | grep -q "$VOLUME_NAME"; then
        print_info "Volume does not exist. Nothing to clean."
        return
    fi
    
    print_error "WARNING: This will DELETE all cached models!"
    print_warning "Are you sure? (type 'yes' to confirm)"
    read -r CONFIRM
    
    if [ "$CONFIRM" != "yes" ]; then
        print_info "Cancelled."
        exit 0
    fi
    
    # Check if volume is in use
    if docker volume ls -q | grep -q "$VOLUME_NAME"; then
        if docker ps -a --format '{{.Mounts}}' | grep -q "$VOLUME_NAME"; then
            print_warning "Stopping containers using this volume..."
            docker-compose -f docker/docker-compose.yml down
        fi
    fi
    
    docker volume rm "$VOLUME_NAME"
    
    if [ $? -eq 0 ]; then
        print_success "Successfully deleted volume. All models removed."
        echo ""
        print_info "Run './auto launch' to recreate and download models again."
    else
        print_error "Failed to delete volume"
        exit 1
    fi
}

# Check disk usage
disk_usage() {
    print_header "Disk Usage"
    
    check_docker
    
    echo "Docker system disk usage:"
    docker system df
    
    echo ""
    echo "Volume information:"
    if docker volume ls | grep -q "open-core-graph-rag_ollama-models"; then
        echo "LLM models volume exists and is taking up space."
        echo ""
        print_info "To see detailed volume info, run: ./manage_models.sh info"
    else
        print_info "No LLM models volume yet."
    fi
}

# Help message
show_help() {
    cat << EOF
${BLUE}LLM Models Management Script${NC}

${GREEN}Usage:${NC}
  ./manage_models.sh <command> [arguments]

${GREEN}Commands:${NC}
  list              List all downloaded models
  info              Show volume storage information
  download <model>  Download a specific model
                    Example: ./manage_models.sh download mistral
  remove <model>    Remove a model to free space
                    Example: ./manage_models.sh remove mistral
  setup             Setup default models (mistral)
  disk              Check disk usage
  clean             Delete volume and all models (dangerous!)
  help              Show this help message

${GREEN}Available Models:${NC}
  mistral         - 5-7GB (recommended, most balanced)
  neural-chat     - 3-5GB (good for chat)
  llama2          - 7GB (general purpose)
  zephyr          - 3-4GB (instruction-following)
  dolphin-mixtral - 45GB (large, for complex tasks)

${GREEN}Examples:${NC}
  # Check what models are downloaded
  ./manage_models.sh list

  # Download mistral (takes ~10 minutes)
  ./manage_models.sh download mistral

  # Add another model for competitive extraction
  ./manage_models.sh download neural-chat

  # Check storage
  ./manage_models.sh disk

${GREEN}Tips:${NC}
  • First launch takes 10-20 minutes (downloads models)
  • Subsequent launches are instant (models cached in volume)
  • Volume persists even after containers stop
  • Models are shared across container restarts
  • For competitive extraction, download 2-3 models

${BLUE}For more info, see: DOCKER_LLM_PERSISTENCE.md${NC}

EOF
}

# Main script
main() {
    COMMAND="${1:-help}"
    
    case "$COMMAND" in
        list)
            list_models
            ;;
        info)
            volume_info
            ;;
        download)
            download_model "$2"
            ;;
        remove)
            remove_model "$2"
            ;;
        setup)
            setup_defaults
            ;;
        disk)
            disk_usage
            ;;
        clean)
            cleanup
            ;;
        help)
            show_help
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
