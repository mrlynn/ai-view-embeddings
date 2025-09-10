import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
} from 'reactflow';
import 'reactflow/dist/style.css';
import ForceGraph3D from '3d-force-graph';
import * as THREE from 'three';
import SpriteText from 'three-spritetext';
import { OpenAIEmbeddings } from './api/OpenAIEmbeddings';
import './App.css';

// Initial sample embedding data with 3D positions
const initialEmbeddingData = [
  // Programming concepts cluster
  { id: 'javascript', label: 'JavaScript', x: 100, y: 100, z: 50, category: 'programming', color: '#3b82f6', vector: [0.23, -0.52, 0.81, -0.14] },
  { id: 'python', label: 'Python', x: 150, y: 120, z: 80, category: 'programming', color: '#3b82f6', vector: [0.31, -0.48, 0.73, -0.21] },
  { id: 'react', label: 'React', x: 80, y: 150, z: 30, category: 'programming', color: '#3b82f6', vector: [0.18, -0.58, 0.76, -0.09] },
  { id: 'nodejs', label: 'Node.js', x: 120, y: 80, z: 60, category: 'programming', color: '#3b82f6', vector: [0.26, -0.55, 0.79, -0.12] },
  { id: 'github', label: 'GitHub', x: 180, y: 140, z: 90, category: 'programming', color: '#3b82f6', vector: [0.35, -0.46, 0.68, -0.25] },
  
  // Animal concepts cluster
  { id: 'cat', label: 'Cat', x: 400, y: 300, z: -50, category: 'animals', color: '#10b981', vector: [0.82, 0.47, 0.21, -0.18] },
  { id: 'dog', label: 'Dog', x: 450, y: 280, z: -80, category: 'animals', color: '#10b981', vector: [0.79, 0.52, 0.18, -0.22] },
  { id: 'tiger', label: 'Tiger', x: 380, y: 350, z: -60, category: 'animals', color: '#10b981', vector: [0.75, 0.42, 0.26, -0.15] },
  { id: 'elephant', label: 'Elephant', x: 500, y: 320, z: -70, category: 'animals', color: '#10b981', vector: [0.68, 0.38, 0.31, -0.12] },
  
  // Food concepts cluster
  { id: 'pizza', label: 'Pizza', x: 250, y: 450, z: 120, category: 'food', color: '#f59e0b', vector: [-0.42, 0.76, -0.35, 0.28] },
  { id: 'pasta', label: 'Pasta', x: 280, y: 420, z: 150, category: 'food', color: '#f59e0b', vector: [-0.39, 0.72, -0.38, 0.32] },
  { id: 'sushi', label: 'Sushi', x: 320, y: 470, z: 140, category: 'food', color: '#f59e0b', vector: [-0.45, 0.68, -0.41, 0.25] },
  { id: 'burger', label: 'Burger', x: 220, y: 480, z: 110, category: 'food', color: '#f59e0b', vector: [-0.38, 0.74, -0.32, 0.36] },
  { id: 'chef', label: 'Chef', x: 300, y: 400, z: 130, category: 'food', color: '#f59e0b', vector: [-0.35, 0.65, -0.45, 0.30] },
  
  // Transportation cluster
  { id: 'car', label: 'Car', x: 600, y: 150, z: -150, category: 'transport', color: '#ef4444', vector: [-0.68, -0.56, -0.21, 0.62] },
  { id: 'airplane', label: 'Airplane', x: 650, y: 120, z: -180, category: 'transport', color: '#ef4444', vector: [-0.72, -0.52, -0.18, 0.58] },
  { id: 'bicycle', label: 'Bicycle', x: 580, y: 200, z: -120, category: 'transport', color: '#ef4444', vector: [-0.65, -0.61, -0.24, 0.65] },
  { id: 'train', label: 'Train', x: 630, y: 180, z: -160, category: 'transport', color: '#ef4444', vector: [-0.70, -0.58, -0.22, 0.61] },
];

// Custom node component for 2D view
const CustomNode = ({ data }) => {
  return (
    <div className="custom-node">
      <div className="node-color" style={{ backgroundColor: data.color }}></div>
      <div className="node-label">{data.label}</div>
      <div className="node-category">{data.category}</div>
      <div className="node-vector">
        {data.vector ? 
          (data.vector.length > 5 ? 
            `[${data.vector.slice(0, 2).map(v => v.toFixed(2)).join(', ')}, ...]` : 
            `[${data.vector.map(v => v.toFixed(2)).join(', ')}]`
          ) : ''
        }
      </div>
    </div>
  );
};

const nodeTypes = {
  custom: CustomNode,
};

export default function EmbeddingVisualizer() {
  const [embeddingData, setEmbeddingData] = useState(initialEmbeddingData);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [showSimilarity, setShowSimilarity] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [distanceMetric, setDistanceMetric] = useState('euclidean');
  const [threshold, setThreshold] = useState(0.7);
  const [is3DView, setIs3DView] = useState(false);
  const [originalNodePositions, setOriginalNodePositions] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [customInput, setCustomInput] = useState('');
  const [showCustomInputForm, setShowCustomInputForm] = useState(false);
  const [showDebugPanel, setShowDebugPanel] = useState(false);
  
  // References
  const graphRef = useRef(null);
  const graph3DInstanceRef = useRef(null);

  // Calculate distance based on different metrics
  const calculateDistance = useCallback((vector1, vector2, metric) => {
    // Helper functions for vector calculations
    const dotProduct = (a, b) => a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitude = vec => Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
    
    switch (metric) {
      case 'euclidean':
        return Math.sqrt(vector1.reduce((sum, val, i) => sum + Math.pow(val - vector2[i], 2), 0));
      case 'cosine':
        const dot = dotProduct(vector1, vector2);
        const magA = magnitude(vector1);
        const magB = magnitude(vector2);
        return 1 - (dot / (magA * magB)); // Cosine distance (1 - cosine similarity)
      case 'dot':
        return 1 - ((dotProduct(vector1, vector2) + 1) / 2); // Normalized to [0,1] range
      default:
        return Math.sqrt(vector1.reduce((sum, val, i) => sum + Math.pow(val - vector2[i], 2), 0));
    }
  }, []);

  // Convert embedding data to ReactFlow nodes
  const initialNodes = useMemo(() => {
    // Create a positions object to store original positions
    const positions = {};
    
    const nodes = embeddingData.map((item) => {
      // Store the original position
      positions[item.id] = { x: item.x, y: item.y, z: item.z || 0 };
      
      return {
        id: item.id,
        type: 'custom',
        position: { x: item.x, y: item.y },
        data: { 
          label: item.label, 
          category: item.category, 
          color: item.color,
          vector: item.vector,
          originalData: item 
        },
      };
    });
    
    // Save the original positions for reset functionality
    setOriginalNodePositions(positions);
    
    return nodes;
  }, [embeddingData]);

  // Return empty edges array - no edge connections in 2D view
  const calculateSimilarityEdges = useCallback(() => {
    console.log("Edge connections disabled in 2D view");
    return [];
  }, []);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Update nodes when embeddingData changes
  useEffect(() => {
    setNodes(initialNodes);
  }, [embeddingData, initialNodes, setNodes]);

  // Filter nodes by category
  const filteredNodes = useMemo(() => {
    if (selectedCategory === 'all') return nodes;
    return nodes.filter(node => node.data.category === selectedCategory);
  }, [nodes, selectedCategory]);

  // Handle similarity toggle - edges disabled in 2D view
  const handleSimilarityToggle = useCallback(() => {
    const newState = !showSimilarity;
    setShowSimilarity(newState);
    
    // Toggle similarity state but don't create edges in 2D view
    setTimeout(() => {
      if (newState) {
        console.log("Edge connections disabled in 2D view");
        // Ensure no edges are shown
        setEdges([]);
        
        // Remove any edge overlay containers that might exist
        const overlayContainer = document.getElementById('edge-overlay-container');
        if (overlayContainer) {
          overlayContainer.remove();
        }
      } else {
        console.log("Hiding similarity connections");
        setEdges([]);
        // Clear selected node when hiding similarity connections
        setSelectedNode(null);
      }
    }, 10);
  }, [showSimilarity, calculateSimilarityEdges, distanceMetric, setEdges]);

  // Simple node click handler - no edge creation
  const onNodeClick = useCallback((_, node) => {
    setSelectedNode(node);
    console.log(`Node clicked: ${node.id}`);
    // No edge creation in 2D view
    setEdges([]);
  }, [setEdges, showSimilarity, distanceMetric]);

  // Function to reset node positions in 2D view
  const handleResetPositions = useCallback(() => {
    setNodes(nds => 
      nds.map(node => ({
        ...node,
        position: originalNodePositions[node.id] || node.position
      }))
    );
    
    // No edges in 2D view
    setEdges([]);
    
    // Clear selected node
    setSelectedNode(null);
  }, [originalNodePositions, setNodes, showSimilarity, calculateSimilarityEdges, distanceMetric, setEdges]);

  // Initialize and update 3D graph
  const initializeGraph3D = useCallback(() => {
    if (!is3DView || !graphRef.current) return;
    
    // Clear any existing graph
    if (graph3DInstanceRef.current) {
      graphRef.current.innerHTML = '';
    }
    
    // Filter nodes by category if needed
    let graphData = { nodes: [], links: [] };
    const filteredData = selectedCategory === 'all' 
      ? embeddingData 
      : embeddingData.filter(item => item.category === selectedCategory);
    
    // Prepare nodes for 3D graph
    graphData.nodes = filteredData.map(item => ({
      id: item.id,
      name: item.label,
      category: item.category,
      color: item.color,
      val: 2, // Node size
      vector: item.vector,
      x: item.x / 4, // Scale down for better 3D view
      y: -item.y / 4, // Invert Y for 3D coordinate system
      z: item.z / 4
    }));
    
    // Add links if similarity is enabled
    if (showSimilarity) {
      // Calculate links based on distance metric
      const links = [];
      const maxDistance = {
        euclidean: 2.0,
        cosine: 1.0,
        dot: 1.0
      };
      
      const thresholdDistance = threshold * maxDistance[distanceMetric];
      
      for (let i = 0; i < filteredData.length; i++) {
        for (let j = i + 1; j < filteredData.length; j++) {
          const item1 = filteredData[i];
          const item2 = filteredData[j];
          
          if (!item1.vector || !item2.vector) continue;
          
          const distance = calculateDistance(item1.vector, item2.vector, distanceMetric);
          
          if (distance < thresholdDistance) {
            const similarity = 1 - (distance / thresholdDistance);
            links.push({
              source: item1.id,
              target: item2.id,
              distance: distance,
              similarity: similarity,
              width: Math.max(1, similarity * 5),
              color: '#94a3b8',
              opacity: 0.6
            });
          }
        }
      }
      
      graphData.links = links;
    }
    
    // Initialize 3D force graph
    const Graph = ForceGraph3D()(graphRef.current)
      .graphData(graphData)
      .nodeLabel(node => `${node.name} (${node.category})`)
      .nodeColor(node => node.color)
      .nodeThreeObject(node => {
        // Create a sphere for the node
        const sphere = new THREE.Mesh(
          new THREE.SphereGeometry(node.val),
          new THREE.MeshLambertMaterial({ color: node.color, transparent: true, opacity: 0.8 })
        );
        
        // Add text label
        const sprite = new SpriteText(node.name);
        sprite.color = '#ffffff';
        sprite.backgroundColor = node.color;
        sprite.padding = 2;
        sprite.textHeight = 3;
        sprite.position.y = 4;
        
        // Create a group to hold both objects
        const group = new THREE.Group();
        group.add(sphere);
        group.add(sprite);
        
        return group;
      })
      .linkWidth(link => link.width)
      .linkColor(link => link.color)
      .linkOpacity(0.6)
      .linkLabel(link => `Distance: ${link.distance.toFixed(3)}`)
      .linkDirectionalParticles(link => link.similarity * 4)
      .linkDirectionalParticleWidth(2)
      .linkDirectionalParticleColor(() => '#ffffff')
      .onNodeClick(node => {
        // Highlight connections on click
        Graph.linkColor(link => 
          (link.source.id === node.id || link.target.id === node.id) ? '#f59e0b' : '#94a3b8'
        );
        Graph.linkWidth(link => 
          (link.source.id === node.id || link.target.id === node.id) ? link.width * 1.5 : link.width
        );
        Graph.linkOpacity(link => 
          (link.source.id === node.id || link.target.id === node.id) ? 1 : 0.3
        );
        
        // Set link label visibility based on connection to selected node
        Graph.linkLabel(link => {
          if (link.source.id === node.id || link.target.id === node.id) {
            return `Distance: ${link.distance.toFixed(3)}`;
          } else {
            return null; // Hide labels for other links
          }
        });
        
        // Update selected node info
        setSelectedNode({
          id: node.id,
          data: {
            label: node.name,
            category: node.category,
            vector: node.vector
          }
        });
      });
    
    // Save reference to the graph instance
    graph3DInstanceRef.current = Graph;
    
    // Adjust camera position for better viewing angle
    Graph.cameraPosition({ x: 0, y: 0, z: 300 });
  }, [is3DView, selectedCategory, showSimilarity, distanceMetric, threshold, embeddingData, calculateDistance]);
  
  
  // Reset 3D graph to original positions
  const resetGraph3D = useCallback(() => {
    if (!is3DView || !graph3DInstanceRef.current) return;
    
    // Re-initialize the graph to reset positions
    initializeGraph3D();
    
    // Clear selected node
    setSelectedNode(null);
  }, [is3DView, initializeGraph3D]);

  // Handle custom text submission for OpenAI embeddings
  const handleCustomTextSubmit = async (e) => {
    e.preventDefault();
    
    if (!customInput.trim()) return;
    
    setIsLoading(true);
    
    try {
      // Process each line as a separate concept
      const concepts = customInput.split('\n').filter(line => line.trim());
      
      if (concepts.length === 0) {
        alert('Please enter at least one concept (one per line)');
        setIsLoading(false);
        return;
      }
      
      // Get embeddings from OpenAI
      const embeddingsApi = new OpenAIEmbeddings();
      const results = await embeddingsApi.getEmbeddings(concepts);
      
      // Define categories and colors for custom concepts
      const categories = ['custom-1', 'custom-2', 'custom-3', 'custom-4'];
      const colors = ['#8b5cf6', '#ec4899', '#14b8a6', '#6366f1'];
      
      // Create new data with random positions but real vectors
      const newData = results.map((item, index) => {
        const categoryIndex = Math.floor(index % 4);
        const randomDistance = 200 + Math.random() * 400;
        const angle = (index / results.length) * Math.PI * 2;
        
        return {
          id: `custom-${index}`,
          label: item.text,
          x: Math.cos(angle) * randomDistance + 400,
          y: Math.sin(angle) * randomDistance + 300,
          z: (Math.random() - 0.5) * 300,
          category: categories[categoryIndex],
          color: colors[categoryIndex],
          vector: item.embedding // Keep full embedding data
        };
      });
      
      // Combine with existing data or replace it
      setEmbeddingData(prev => [...prev, ...newData]);
      setShowCustomInputForm(false);
      setCustomInput('');
      
      // Reset UI
      setShowSimilarity(false);
      setEdges([]);
      setSelectedNode(null);
      
    } catch (error) {
      console.error("Error getting embeddings:", error);
      alert(`Error getting embeddings: ${error.message || 'Unknown error'}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Categories including custom ones if they exist
  const categories = useMemo(() => {
    const uniqueCategories = [...new Set(embeddingData.map(item => item.category))];
    return ['all', ...uniqueCategories];
  }, [embeddingData]);

  // Update 3D graph when view mode changes
  useEffect(() => {
    if (is3DView) {
      initializeGraph3D();
    }
  }, [is3DView, initializeGraph3D]);

  return (
    <div className="app-container">
      {/* Control Panel */}
      <div className="control-panel">
        <h2 className="control-title">AI Embeddings Visualizer</h2>
        
        <div className="control-group">
          <label className="control-label">View Mode:</label>
          <div className="view-toggle">
            <button
              onClick={() => setIs3DView(false)}
              className={`control-button ${!is3DView ? 'active' : ''}`}
            >
              2D
            </button>
            <button
              onClick={() => {
                setIs3DView(true);
                // Initialize 3D graph after state update
                setTimeout(initializeGraph3D, 0);
              }}
              className={`control-button ${is3DView ? 'active' : ''}`}
            >
              3D
            </button>
          </div>
        </div>
        
        <div className="control-group">
          <label className="control-label">Category Filter:</label>
          <select 
            value={selectedCategory} 
            onChange={(e) => {
              setSelectedCategory(e.target.value);
              if (is3DView) {
                setTimeout(initializeGraph3D, 0);
              }
            }}
            className="control-select"
          >
            {categories.map(cat => (
              <option key={cat} value={cat}>
                {cat.charAt(0).toUpperCase() + cat.slice(1)}
              </option>
            ))}
          </select>
        </div>
        
        <div className="control-group">
          <label className="control-label">Distance Metric:</label>
          <select 
            value={distanceMetric} 
            onChange={(e) => {
              setDistanceMetric(e.target.value);
              if (showSimilarity) {
                if (is3DView) {
                  setTimeout(initializeGraph3D, 0);
                } else {
                  setEdges(calculateSimilarityEdges(e.target.value));
                }
              }
            }}
            className="control-select"
          >
            <option value="euclidean">Euclidean Distance</option>
            <option value="cosine">Cosine Similarity</option>
            <option value="dot">Dot Product</option>
          </select>
        </div>

        <div className="control-group">
          <label className="control-label">Threshold: {threshold.toFixed(2)}</label>
          <input 
            type="range" 
            min="0.1" 
            max="1.0" 
            step="0.05"
            value={threshold}
            onChange={(e) => {
              const newThreshold = parseFloat(e.target.value);
              setThreshold(newThreshold);
              if (showSimilarity) {
                if (is3DView) {
                  setTimeout(initializeGraph3D, 0);
                } else {
                  setEdges(calculateSimilarityEdges(distanceMetric));
                }
              }
            }}
            className="control-range"
          />
        </div>
        
        <div className="control-group">
          <label className="control-checkbox">
            <input
              type="checkbox"
              checked={showSimilarity}
              onChange={() => {
                const newValue = !showSimilarity;
                setShowSimilarity(newValue);
                if (is3DView) {
                  setTimeout(initializeGraph3D, 0);
                } else {
                  handleSimilarityToggle();
                }
              }}
            />
            <span className="checkbox-label">Show Similarity Connections</span>
          </label>
        </div>

        <div className="control-group">
          <button 
            onClick={() => {
              if (is3DView) {
                resetGraph3D();
              } else {
                handleResetPositions();
              }
            }}
            className="control-button"
            disabled={isLoading}
          >
            Reset Positions
          </button>
        </div>
        
        <div className="control-group">
          <button 
            onClick={() => setShowCustomInputForm(!showCustomInputForm)}
            className="control-button add-concepts"
            disabled={isLoading}
          >
            {showCustomInputForm ? 'Cancel' : 'Add Custom Concepts'}
          </button>
        </div>
        
        <div className="control-group">
          <label className="control-checkbox">
            <input
              type="checkbox"
              checked={showDebugPanel}
              onChange={() => setShowDebugPanel(!showDebugPanel)}
            />
            <span className="checkbox-label">Show Edge Debug Panel</span>
          </label>
        </div>
        
        {showCustomInputForm && (
          <form onSubmit={handleCustomTextSubmit} className="custom-input-form">
            <textarea
              value={customInput}
              onChange={(e) => setCustomInput(e.target.value)}
              placeholder="Enter concepts (one per line)"
              className="custom-textarea"
              rows={5}
              disabled={isLoading}
            />
            <button 
              type="submit" 
              className="control-button submit"
              disabled={isLoading}
            >
              {isLoading ? 'Processing...' : 'Get Embeddings'}
            </button>
          </form>
        )}
      </div>

      {/* Info Panel */}
      <div className="info-panel">
        <h3 className="info-title">Understanding Embeddings</h3>
        <p className="info-text">
          Embeddings represent concepts as points in vector space. Similar concepts cluster together.
        </p>
        <ul className="info-list">
          <li>Distance = Semantic similarity</li>
          <li>Clusters = Related concepts</li>
          <li>Click nodes to explore connections</li>
        </ul>
        
        <div className="metric-explanation">
          <h4 className="metric-title">Distance Metrics:</h4>
          <div className="metric-details">
            <p>
              <strong>Euclidean:</strong> Physical distance between points in space.
            </p>
            <p>
              <strong>Cosine:</strong> Angle between vectors (direction similarity).
            </p>
            <p>
              <strong>Dot Product:</strong> Product of magnitudes and cosine of angle.
            </p>
          </div>
        </div>
        
        {selectedNode && (
          <div className="selected-node-info">
            <h4 className="selected-node-title">Selected:</h4>
            <p className="selected-node-details">
              <strong>{selectedNode.data.label}</strong><br/>
              Category: {selectedNode.data.category}<br/>
              <div className="vector-container">
                <small>Vector ({selectedNode.data.vector ? selectedNode.data.vector.length : 0} dimensions): </small>
                <span className="vector-display">{selectedNode.data.vector ? 
                  (selectedNode.data.vector.length > 5 ? 
                    `[${selectedNode.data.vector.slice(0, 3).map(v => v.toFixed(2)).join(', ')}, ...]` : 
                    `[${selectedNode.data.vector.map(v => v.toFixed(2)).join(', ')}]`
                  ) : ''
                }</span>
              </div>
            </p>
            
            {showSimilarity && !is3DView && (
              <div className="distance-info">
                <h5 className="distance-title">Distances ({distanceMetric}):</h5>
                <ul className="distance-list">
                  {edges
                    .filter(edge => edge.source === selectedNode.id || edge.target === selectedNode.id)
                    .map(edge => {
                      const connectedId = edge.source === selectedNode.id ? edge.target : edge.source;
                      const connectedNode = nodes.find(n => n.id === connectedId);
                      return {
                        id: edge.id,
                        label: connectedNode.data.label,
                        distance: parseFloat(edge.data.distance)
                      };
                    })
                    .sort((a, b) => a.distance - b.distance)
                    .map(item => (
                      <li key={item.id}>
                        {item.label}: {item.distance.toFixed(3)}
                      </li>
                    ))}
                </ul>
              </div>
            )}
            
            {showSimilarity && is3DView && graph3DInstanceRef.current && (
              <div className="distance-info">
                <h5 className="distance-title">Distances ({distanceMetric}):</h5>
                <ul className="distance-list">
                  {graph3DInstanceRef.current.graphData().links
                    .filter(link => link.source.id === selectedNode.id || link.target.id === selectedNode.id)
                    .map((link, idx) => {
                      // Get the connected node
                      const connectedNode = link.source.id === selectedNode.id ? link.target : link.source;
                      return {
                        id: `3d-link-${idx}`,
                        label: connectedNode.name,
                        distance: link.distance
                      };
                    })
                    .sort((a, b) => a.distance - b.distance)
                    .map(item => (
                      <li key={item.id}>
                        {item.label}: {item.distance.toFixed(3)}
                      </li>
                    ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="legend">
        <h4 className="legend-title">Categories:</h4>
        <div className="legend-items">
          {categories.filter(cat => cat !== 'all').map(category => {
            const exampleItem = embeddingData.find(item => item.category === category);
            return (
              <div key={category} className="legend-item">
                <div 
                  className="legend-color" 
                  style={{ backgroundColor: exampleItem?.color || '#999' }}
                ></div>
                <span className="legend-text">
                  {category.charAt(0).toUpperCase() + category.slice(1)}
                </span>
              </div>
            );
          })}
        </div>
        
        <div className="current-metric">
          <h4 className="metric-label">Current Metric:</h4>
          <div className="metric-value">
            <strong>{distanceMetric.charAt(0).toUpperCase() + distanceMetric.slice(1)}</strong>
            <div className="threshold-value">
              Threshold: {threshold.toFixed(2)}
            </div>
          </div>
        </div>
      </div>


      {/* Debug Panel */}
      {showDebugPanel && (
        <div className="debug-panel">
          <h3 className="debug-title">Edge Debug Info</h3>
          <div className="debug-content">
            <p><strong>Total Edges:</strong> {edges.length}</p>
            <p><strong>Show Similarity:</strong> {showSimilarity ? 'Yes' : 'No'}</p>
            <p><strong>Edge Connections:</strong> Disabled in 2D view (only available in 3D view)</p>
            
            <div className="edge-list">
              <h4>Edge Details:</h4>
              <p className="no-edges">Edge connections are disabled in 2D view. Switch to 3D view to see edge connections.</p>
            </div>
          </div>
        </div>
      )}

      {/* Visualization Area */}
      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <div className="loading-text">Processing with OpenAI...</div>
        </div>
      )}

      {/* Visualization - either 2D ReactFlow or 3D ForceGraph */}
      {!is3DView ? (
        <ReactFlow
          nodes={filteredNodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          onPaneClick={() => {
            // Reset edges when clicking on the background if showing node connections
            if (selectedNode && !showSimilarity) {
              setEdges([]);
              setSelectedNode(null);
            }
          }}
          nodeTypes={nodeTypes}
          fitView={false}
          attributionPosition="bottom-right"
          defaultEdgeOptions={{
            type: 'straight',
            animated: true,
            style: { stroke: '#ff0000', strokeWidth: 3 }
          }}
        >
          <Background color="#e2e8f0" gap={20} />
          <Controls />
          <MiniMap 
            nodeColor={(node) => node.data.color}
            nodeStrokeWidth={3}
            zoomable
            pannable
          />
        </ReactFlow>
      ) : (
        <div 
          ref={graphRef} 
          className="graph3d-container"
        />
      )}
    </div>
  );
}