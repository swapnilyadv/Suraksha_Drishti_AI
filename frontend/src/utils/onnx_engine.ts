import * as ort from "onnxruntime-web";

// Labels matching the training script
export const ACTION_LABELS: Record<number, string> = {
  0: "Normal",
  1: "Harassment",
  2: "Weapon",
  3: "Male Faces",
  4: "Female Faces",
};

export class SurakshaAI {
  private session: ort.InferenceSession | null = null;

  async loadModel(modelPath: string = "/models/suraksha_vision.onnx") {
    try {
      this.session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ["webgl"], // Use GPU acceleration if available
        graphOptimizationLevel: "all",
      });
      console.log("✅ Suraksha AI Model Loaded");
    } catch (e) {
      console.error("❌ Failed to load ONNX model:", e);
      // Fallback to WASM if WebGL fails
      this.session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ["wasm"],
      });
    }
  }

  async predict(frameSequence: Float32Array[]): Promise<{ label: string; confidence: number; counts: any }> {
    if (!this.session) throw new Error("Model not loaded");

    // Input shape: [1, 16, 3, 112, 112]
    // We expect 16 frames in frameSequence, each pre-processed
    const flatData = new Float32Array(1 * 16 * 3 * 112 * 112);
    
    // Flatten the sequence into the buffer
    let offset = 0;
    for (const frame of frameSequence) {
      flatData.set(frame, offset);
      offset += frame.length;
    }

    const inputTensor = new ort.Tensor("float32", flatData, [1, 16, 3, 112, 112]);
    const feeds = { input: inputTensor };

    const results = await this.session.run(feeds);
    const output = results.output.data as Float32Array;

    // Get the highest probability
    let maxIdx = 0;
    let maxVal = -Infinity;
    for (let i = 0; i < output.length; i++) {
      if (output[i] > maxVal) {
        maxVal = output[i];
        maxIdx = i;
      }
    }

    // Softmax-like confidence
    const expSum = Array.from(output).reduce((a, b) => a + Math.exp(b), 0);
    const confidence = Math.exp(maxVal) / expSum;

    return {
      label: ACTION_LABELS[maxIdx] || "Unknown",
      confidence: confidence,
      counts: {
        male: maxIdx === 3 ? 1 : 0, // Simple mock counting from classification
        female: maxIdx === 4 ? 1 : 0,
        weapon: maxIdx === 2,
      }
    };
  }
}

export const surakshaAI = new SurakshaAI();
