#!/bin/bash

echo "🔄 Restarting F2 Document Processing Services in Docker..."
echo "=============================================="

# Stop and remove existing containers
echo "📛 Stopping existing containers..."
docker-compose -f docker-compose-full.yml down

# Remove old images (optional - comment out if you want to keep them)
echo "🗑️  Removing old images..."
docker-compose -f docker-compose-full.yml rm -f

# Build new images
echo "🔨 Building Docker images..."
docker-compose -f docker-compose-full.yml build --no-cache

# Start services
echo "🚀 Starting services..."
docker-compose -f docker-compose-full.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."
sleep 10

# Check service health
echo "🏥 Checking service health..."
docker-compose -f docker-compose-full.yml ps

# Show logs
echo ""
echo "📋 Service URLs:"
echo "  - API Gateway:               http://localhost:8000"
echo "  - API Gateway Docs:          http://localhost:8000/docs"
echo "  - Classification Service:    http://localhost:8001"
echo "  - Quality Service:           http://localhost:8002"
echo "  - Preprocessing Service:     http://localhost:8003"
echo "  - Entity Extraction Service: http://localhost:8004"
echo "  - Frontend:                  http://localhost:8080"
echo ""
echo "📊 To view logs: docker-compose -f docker-compose-full.yml logs -f"
echo "🛑 To stop services: docker-compose -f docker-compose-full.yml down"
echo ""
echo "✅ Services restarted successfully!"