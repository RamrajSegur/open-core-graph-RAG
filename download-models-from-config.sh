#!/bin/bash
# Script to download LLM models from configuration file
# Reads .ollama-models.conf and downloads all non-commented models
# Used for automated model setup in competitive NER pipeline

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

# Check if Ollama container is running
check_ollama_container() {
    if ! docker ps | grep -q "open-core-graph-rag-ollama"; then
        print_error "Ollama container is not running."
        echo ""
        print_info "Start it with: ./auto launch"
        exit 1
    fi
}

# Check if config file exists
check_config_file() {
    CONFIG_FILE=".ollama-models.conf"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Configuration file '$CONFIG_FILE' not found!"
        exit 1
    fi
}

# Parse config file and extract model names
# Filters out comments and empty lines
get_models_from_config() {
    CONFIG_FILE=".ollama-models.conf"
    
    # Extract non-empty, non-comment lines
    grep -v '^#' "$CONFIG_FILE" | grep -v '^[[:space:]]*$' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

# Download models from config file
download_models_from_config() {
    print_header "Download Models from Configuration"
    
    check_docker
    check_config_file
    check_ollama_container
    
    # Get models from config
    MODELS=$(get_models_from_config)
    
    if [ -z "$MODELS" ]; then
        print_warning "No models configured in .ollama-models.conf"
        echo "Please uncomment models in the configuration file."
        exit 1
    fi
    
    print_info "Models to download from .ollama-models.conf:"
    echo ""
    echo "$MODELS" | nl
    echo ""
    
    print_warning "This will download the specified models (may take 10-60+ minutes)"
    print_info "Download time depends on model size and internet speed"
    echo ""
    
    # Confirm before downloading
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Download cancelled."
        exit 0
    fi
    
    # Download each model
    TOTAL_MODELS=$(echo "$MODELS" | wc -l)
    CURRENT=0
    FAILED_MODELS=()
    DOWNLOADED_MODELS=()
    
    echo "$MODELS" | while read -r MODEL; do
        CURRENT=$((CURRENT + 1))
        
        print_info "[$CURRENT/$TOTAL_MODELS] Downloading: $MODEL"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        
        if docker exec open-core-graph-rag-ollama ollama pull "$MODEL"; then
            print_success "Downloaded: $MODEL"
            DOWNLOADED_MODELS+=("$MODEL")
        else
            print_error "Failed to download: $MODEL"
            FAILED_MODELS+=("$MODEL")
        fi
        
        echo ""
    done
    
    # Summary
    print_header "Download Summary"
    
    if [ ${#DOWNLOADED_MODELS[@]} -gt 0 ]; then
        echo "Successfully downloaded models:"
        for model in "${DOWNLOADED_MODELS[@]}"; do
            echo -e "${GREEN}  ✓${NC} $model"
        done
    fi
    
    if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
        echo ""
        echo "Failed to download models:"
        for model in "${FAILED_MODELS[@]}"; do
            echo -e "${RED}  ✗${NC} $model"
        done
        exit 1
    fi
    
    print_success "All models downloaded successfully!"
    print_info "Models are now cached and will persist across container restarts."
    echo ""
    print_info "View downloaded models with: ./manage_models.sh list"
}

# Show models that would be downloaded
show_models_in_config() {
    print_header "Models in Configuration File"
    
    check_config_file
    
    MODELS=$(get_models_from_config)
    
    if [ -z "$MODELS" ]; then
        print_warning "No models configured in .ollama-models.conf"
        exit 1
    fi
    
    echo "Models that will be downloaded:"
    echo "$MODELS" | nl
}

# Main script
main() {
    case "${1:-download}" in
        download)
            download_models_from_config
            ;;
        list)
            show_models_in_config
            ;;
        help)
            echo "Usage: $0 [COMMAND]"
            echo ""
            echo "Commands:"
            echo "  download    Download all models from .ollama-models.conf (default)"
            echo "  list        Show models configured in .ollama-models.conf"
            echo "  help        Show this help message"
            echo ""
            echo "Configuration file: .ollama-models.conf"
            echo ""
            echo "Examples:"
            echo "  $0                  # Download all configured models"
            echo "  $0 download         # Same as above"
            echo "  $0 list             # Show models that will be downloaded"
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Run '$0 help' for usage information."
            exit 1
            ;;
    esac
}

main "$@"
