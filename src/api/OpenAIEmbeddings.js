import OpenAI from 'openai';

export class OpenAIEmbeddings {
  constructor() {
    // Initialize OpenAI with API key from environment variable
    this.openai = new OpenAI({
      apiKey: process.env.REACT_APP_OPENAI_API_KEY,
      dangerouslyAllowBrowser: true, // Allow client-side usage (consider security implications)
    });
    
    // Default embedding model
    this.model = "text-embedding-3-small";
  }

  /**
   * Get embeddings for a list of text items
   * @param {string[]} texts - Array of text items to generate embeddings for
   * @returns {Promise<Array<{text: string, embedding: number[]}>>} - Array of objects with text and its embedding
   */
  async getEmbeddings(texts) {
    if (!Array.isArray(texts) || texts.length === 0) {
      throw new Error("Input must be a non-empty array of strings");
    }

    try {
      // Make batch request to OpenAI API
      const response = await this.openai.embeddings.create({
        model: this.model,
        input: texts,
      });

      // Extract embeddings and pair with original texts
      const results = response.data.map((item, index) => ({
        text: texts[index],
        embedding: item.embedding,
      }));

      return results;
    } catch (error) {
      console.error("Error generating embeddings:", error);
      throw new Error(`Failed to generate embeddings: ${error.message}`);
    }
  }

  /**
   * Calculate similarity between two vectors using cosine similarity
   * @param {number[]} vec1 - First vector
   * @param {number[]} vec2 - Second vector
   * @returns {number} - Cosine similarity (between -1 and 1)
   */
  static calculateCosineSimilarity(vec1, vec2) {
    if (vec1.length !== vec2.length) {
      throw new Error("Vectors must be of the same length");
    }
    
    // Calculate dot product
    const dotProduct = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
    
    // Calculate magnitudes
    const mag1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
    const mag2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));
    
    // Calculate cosine similarity
    return dotProduct / (mag1 * mag2);
  }
}

export default OpenAIEmbeddings;