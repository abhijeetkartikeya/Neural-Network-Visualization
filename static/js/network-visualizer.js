// Network Visualizer
class NetworkVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.animationFrame = null;
        this.particles = [];
        this.layers = [4, 6, 6, 3]; // Default architecture

        this.setupCanvas();
        this.animate();
    }

    setupCanvas() {
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }

    setArchitecture(layers) {
        this.layers = layers;
        this.particles = [];
    }

    drawNetwork() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear with fade effect
        ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
        ctx.fillRect(0, 0, width, height);

        const layerSpacing = width / (this.layers.length + 1);
        const maxNeurons = Math.max(...this.layers);

        // Draw connections first
        for (let i = 0; i < this.layers.length - 1; i++) {
            const x1 = layerSpacing * (i + 1);
            const x2 = layerSpacing * (i + 2);

            const neurons1 = this.layers[i];
            const neurons2 = this.layers[i + 1];

            const spacing1 = height / (neurons1 + 1);
            const spacing2 = height / (neurons2 + 1);

            for (let j = 0; j < neurons1; j++) {
                const y1 = spacing1 * (j + 1);

                for (let k = 0; k < neurons2; k++) {
                    const y2 = spacing2 * (k + 1);

                    ctx.strokeStyle = 'rgba(139, 92, 246, 0.2)';
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(x1, y1);
                    ctx.lineTo(x2, y2);
                    ctx.stroke();
                }
            }
        }

        // Draw neurons
        for (let i = 0; i < this.layers.length; i++) {
            const x = layerSpacing * (i + 1);
            const neurons = this.layers[i];
            const spacing = height / (neurons + 1);

            for (let j = 0; j < neurons; j++) {
                const y = spacing * (j + 1);

                // Neuron glow
                const gradient = ctx.createRadialGradient(x, y, 0, x, y, 15);
                if (i === 0) {
                    gradient.addColorStop(0, 'rgba(0, 212, 255, 0.8)');
                    gradient.addColorStop(1, 'rgba(0, 212, 255, 0)');
                } else if (i === this.layers.length - 1) {
                    gradient.addColorStop(0, 'rgba(255, 0, 255, 0.8)');
                    gradient.addColorStop(1, 'rgba(255, 0, 255, 0)');
                } else {
                    gradient.addColorStop(0, 'rgba(139, 92, 246, 0.8)');
                    gradient.addColorStop(1, 'rgba(139, 92, 246, 0)');
                }

                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(x, y, 15, 0, Math.PI * 2);
                ctx.fill();

                // Neuron core
                ctx.fillStyle = i === 0 ? '#00d4ff' :
                    i === this.layers.length - 1 ? '#ff00ff' : '#8b5cf6';
                ctx.beginPath();
                ctx.arc(x, y, 8, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Draw and update particles
        this.updateParticles();
    }

    updateParticles() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const layerSpacing = width / (this.layers.length + 1);

        // Add new particles occasionally
        if (Math.random() < 0.1 && this.particles.length < 20) {
            const neurons = this.layers[0];
            const spacing = height / (neurons + 1);
            const neuronIndex = Math.floor(Math.random() * neurons);

            this.particles.push({
                x: layerSpacing,
                y: spacing * (neuronIndex + 1),
                layer: 0,
                neuron: neuronIndex,
                progress: 0,
                targetNeuron: Math.floor(Math.random() * this.layers[1])
            });
        }

        // Update and draw particles
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const particle = this.particles[i];

            if (particle.layer >= this.layers.length - 1) {
                this.particles.splice(i, 1);
                continue;
            }

            const x1 = layerSpacing * (particle.layer + 1);
            const x2 = layerSpacing * (particle.layer + 2);

            const spacing1 = height / (this.layers[particle.layer] + 1);
            const spacing2 = height / (this.layers[particle.layer + 1] + 1);

            const y1 = spacing1 * (particle.neuron + 1);
            const y2 = spacing2 * (particle.targetNeuron + 1);

            particle.progress += 0.02;

            if (particle.progress >= 1) {
                particle.layer++;
                particle.neuron = particle.targetNeuron;
                particle.progress = 0;

                if (particle.layer < this.layers.length - 1) {
                    particle.targetNeuron = Math.floor(Math.random() * this.layers[particle.layer + 1]);
                }
            }

            const x = x1 + (x2 - x1) * particle.progress;
            const y = y1 + (y2 - y1) * particle.progress;

            // Draw particle
            const gradient = ctx.createRadialGradient(x, y, 0, x, y, 8);
            gradient.addColorStop(0, 'rgba(0, 212, 255, 1)');
            gradient.addColorStop(1, 'rgba(0, 212, 255, 0)');

            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = '#00d4ff';
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    animate() {
        this.drawNetwork();
        this.animationFrame = requestAnimationFrame(() => this.animate());
    }

    stop() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }
}

// Initialize network visualizer
let networkVisualizer;

document.addEventListener('DOMContentLoaded', () => {
    networkVisualizer = new NetworkVisualizer('network-canvas');
});
