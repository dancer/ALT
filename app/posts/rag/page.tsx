"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { ChevronUp, Clock, Share2, PlayCircle, Check } from "lucide-react";

export default function LocalAgenticRAG() {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const totalHeight =
        document.documentElement.scrollHeight -
        document.documentElement.clientHeight;
      const progress = (window.scrollY / totalHeight) * 100;
      setScrollProgress(progress);
      setShowScrollTop(window.scrollY > 300);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleShare = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy: ", err);
    }
  };

  return (
    <article className="relative space-y-6 text-xs">
      <div className="fixed top-0 left-0 w-full h-1 bg-gray-200">
        <div
          className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
          style={{ width: `${scrollProgress}%` }}
        ></div>
      </div>

      <Link
        href="/"
        className="text-blue-400 hover:underline inline-block mb-4"
      >
        {"<"} Back to home
      </Link>

      <header className="space-y-2">
        <h1 className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-500">
          Local Agentic RAG: Revolutionizing LLMs with Private Knowledge
        </h1>
        <p className="text-gray-400">
          Unlock the full potential of Large Language Models with advanced
          retrieval techniques and AI agents
        </p>
        <div className="flex items-center space-x-4 text-gray-500">
          <span className="flex items-center">
            <Clock size={12} className="mr-1" /> 10 min read
          </span>
          <button
            onClick={handleShare}
            className="flex items-center hover:text-gray-300 transition-colors"
          >
            {copied ? (
              <Check size={12} className="mr-1" />
            ) : (
              <Share2 size={12} className="mr-1" />
            )}
            {copied ? "Copied!" : "Share"}
          </button>
        </div>
      </header>

      <nav className="border border-gray-800 rounded p-4">
        <h2 className="font-semibold mb-2">Table of Contents</h2>
        <ul className="space-y-1">
          <li>
            <a
              href="#introduction"
              className="hover:text-blue-400 transition-colors"
            >
              Introduction
            </a>
          </li>
          <li>
            <a
              href="#evolution"
              className="hover:text-blue-400 transition-colors"
            >
              Evolution of Knowledge Integration
            </a>
          </li>
          <li>
            <a
              href="#limitations"
              className="hover:text-blue-400 transition-colors"
            >
              Limitations of Simple RAG
            </a>
          </li>
          <li>
            <a
              href="#deep-dive"
              className="hover:text-blue-400 transition-colors"
            >
              Local Agentic RAG: A Deep Dive
            </a>
          </li>
          <li>
            <a
              href="#pipeline"
              className="hover:text-blue-400 transition-colors"
            >
              The Agentic RAG Pipeline
            </a>
          </li>
          <li>
            <a
              href="#implementation"
              className="hover:text-blue-400 transition-colors"
            >
              Implementing Local Agentic RAG
            </a>
          </li>
          <li>
            <a
              href="#conclusion"
              className="hover:text-blue-400 transition-colors"
            >
              Conclusion
            </a>
          </li>
          <li>
            <a
              href="#video-resources"
              className="hover:text-blue-400 transition-colors"
            >
              Video Resources
            </a>
          </li>
        </ul>
      </nav>

      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 800 400"
        className="w-full h-auto"
      >
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="0"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#ffffff" />
          </marker>
        </defs>

        <rect width="100%" height="100%" fill="#000000" />

        <text
          x="400"
          y="30"
          fontSize="24"
          fill="#ffffff"
          textAnchor="middle"
          fontWeight="bold"
        >
          Local Agentic RAG System
        </text>

        <rect
          x="50"
          y="60"
          width="120"
          height="60"
          rx="10"
          stroke="#ffffff"
          strokeWidth="2"
          fill="none"
        />
        <text x="110" y="95" fontSize="14" fill="#ffffff" textAnchor="middle">
          User Query
        </text>

        <rect
          x="250"
          y="60"
          width="140"
          height="60"
          rx="10"
          stroke="#ffffff"
          strokeWidth="2"
          fill="none"
        />
        <text x="320" y="85" fontSize="14" fill="#ffffff" textAnchor="middle">
          Query Translation
        </text>
        <text x="320" y="105" fontSize="14" fill="#ffffff" textAnchor="middle">
          Agent
        </text>

        <rect
          x="470"
          y="60"
          width="120"
          height="60"
          rx="10"
          stroke="#ffffff"
          strokeWidth="2"
          fill="none"
        />
        <text x="530" y="95" fontSize="14" fill="#ffffff" textAnchor="middle">
          Hybrid Search
        </text>

        <rect
          x="670"
          y="60"
          width="120"
          height="60"
          rx="10"
          stroke="#ffffff"
          strokeWidth="2"
          fill="none"
        />
        <text x="730" y="85" fontSize="14" fill="#ffffff" textAnchor="middle">
          Metadata
        </text>
        <text x="730" y="105" fontSize="14" fill="#ffffff" textAnchor="middle">
          Filtering Agent
        </text>

        <rect
          x="470"
          y="180"
          width="120"
          height="60"
          rx="10"
          stroke="#ffffff"
          strokeWidth="2"
          fill="none"
        />
        <text x="530" y="215" fontSize="14" fill="#ffffff" textAnchor="middle">
          Vector DB
        </text>

        <rect
          x="670"
          y="180"
          width="120"
          height="60"
          rx="10"
          stroke="#ffffff"
          strokeWidth="2"
          fill="none"
        />
        <text x="730" y="215" fontSize="14" fill="#ffffff" textAnchor="middle">
          Keyword Index
        </text>

        <rect
          x="250"
          y="300"
          width="140"
          height="60"
          rx="10"
          stroke="#ffffff"
          strokeWidth="2"
          fill="none"
        />
        <text x="320" y="325" fontSize="14" fill="#ffffff" textAnchor="middle">
          Corrective RAG
        </text>
        <text x="320" y="345" fontSize="14" fill="#ffffff" textAnchor="middle">
          Agent
        </text>

        <rect
          x="50"
          y="300"
          width="120"
          height="60"
          rx="10"
          stroke="#ffffff"
          strokeWidth="2"
          fill="none"
        />
        <text x="110" y="335" fontSize="14" fill="#ffffff" textAnchor="middle">
          Response
        </text>

        <line
          x1="170"
          y1="90"
          x2="250"
          y2="90"
          stroke="#ffffff"
          strokeWidth="2"
          markerEnd="url(#arrowhead)"
        />
        <line
          x1="390"
          y1="90"
          x2="470"
          y2="90"
          stroke="#ffffff"
          strokeWidth="2"
          markerEnd="url(#arrowhead)"
        />
        <line
          x1="590"
          y1="90"
          x2="670"
          y2="90"
          stroke="#ffffff"
          strokeWidth="2"
          markerEnd="url(#arrowhead)"
        />
        <line
          x1="530"
          y1="120"
          x2="530"
          y2="180"
          stroke="#ffffff"
          strokeWidth="2"
          markerEnd="url(#arrowhead)"
        />
        <line
          x1="730"
          y1="120"
          x2="730"
          y2="180"
          stroke="#ffffff"
          strokeWidth="2"
          markerEnd="url(#arrowhead)"
        />
        <path
          d="M 730 240 C 730 270, 320 270, 320 300"
          stroke="#ffffff"
          fill="none"
          strokeWidth="2"
          markerEnd="url(#arrowhead)"
        />
        <path
          d="M 530 240 C 530 270, 320 270, 320 300"
          stroke="#ffffff"
          fill="none"
          strokeWidth="2"
          markerEnd="url(#arrowhead)"
        />
        <line
          x1="250"
          y1="330"
          x2="170"
          y2="330"
          stroke="#ffffff"
          strokeWidth="2"
          markerEnd="url(#arrowhead)"
        />
      </svg>

      <section id="introduction" className="space-y-4">
        <h2 className="text-sm font-semibold">
          Introduction: The Promise of 10x Performance
        </h2>
        <p>
          In the rapidly evolving landscape of artificial intelligence, Large
          Language Models (LLMs) have emerged as powerful tools for managing and
          interpreting vast amounts of information. However, a significant
          challenge remains: how can we make these models perform exceptionally
          well with private, organization-specific knowledge?
        </p>
        <p>
          Enter Local Agentic RAG (Retrieval Augmented Generation) - a
          cutting-edge approach that promises to make LLMs perform "10x better
          with private knowledge". This isn't just an incremental improvement;
          it's a paradigm shift in how we leverage AI for information retrieval
          and processing within organizations.
        </p>
        <blockquote className="border-l-2 border-purple-500 pl-4 italic text-gray-400">
          "I want Llama3 to perform 10x with my private knowledge" - The driving
          vision behind Local Agentic RAG
        </blockquote>
      </section>

      <section id="evolution" className="space-y-4">
        <h2 className="text-sm font-semibold">
          The Evolution of Knowledge Integration in LLMs
        </h2>
        <p>
          To appreciate the revolutionary nature of Local Agentic RAG, let's
          first understand the traditional methods of integrating private
          knowledge into LLMs:
        </p>
        <div className="space-y-2">
          <div className="flex items-start">
            <span className="inline-block w-2 h-2 rounded-full bg-blue-500 mt-1.5 mr-2"></span>
            <div>
              <h3 className="font-semibold">1. Fine-tuning</h3>
              <p>
                <strong>Pros:</strong> Fast inference, deep integration of
                knowledge
              </p>
              <p>
                <strong>Cons:</strong> Requires expertise, time-consuming,
                potential for catastrophic forgetting
              </p>
            </div>
          </div>
          <div className="flex items-start">
            <span className="inline-block w-2 h-2 rounded-full bg-purple-500 mt-1.5 mr-2"></span>
            <div>
              <h3 className="font-semibold">2. Simple RAG</h3>
              <p>
                <strong>Pros:</strong> Easier to implement, flexible knowledge
                base
              </p>
              <p>
                <strong>Cons:</strong> Can be slow, may struggle with complex
                queries
              </p>
            </div>
          </div>
        </div>
      </section>

      <section id="limitations" className="space-y-4">
        <h2 className="text-sm font-semibold">The Limitations of Simple RAG</h2>
        <p>
          While Simple RAG represented a significant step forward, it still
          faces several challenges when dealing with real-world, messy data:
        </p>
        <ul className="list-disc list-inside space-y-2">
          <li>
            Difficulty in processing non-textual data (e.g., images, tables,
            code snippets)
          </li>
          <li>
            Inconsistent performance across different data types and query
            complexities
          </li>
          <li>
            Inability to handle multi-hop reasoning or questions requiring
            synthesized information from multiple sources
          </li>
          <li>Lack of dynamic adaptation to user intent and context</li>
        </ul>
        <p>
          These limitations set the stage for the next evolution in knowledge
          integration: Local Agentic RAG.
        </p>
      </section>

      <section id="deep-dive" className="space-y-4">
        <h2 className="text-sm font-semibold">
          Local Agentic RAG: A Deep Dive
        </h2>
        <p>
          Local Agentic RAG represents a quantum leap in how we approach
          knowledge retrieval and integration in LLMs. By incorporating AI
          agents into the RAG process, we can create a more dynamic,
          intelligent, and context-aware system.
        </p>

        <h3 className="font-semibold">Key Components of Local Agentic RAG</h3>

        <div className="space-y-4">
          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h4 className="font-semibold">
              1. Advanced Parsing with Llama Parts and Fire Craw
            </h4>
            <p>
              Llama Parts and Fire Craw are cutting-edge parsers that transform
              complex documents and web data into LLM-friendly formats:
            </p>
            <ul className="list-disc list-inside">
              <li>
                Llama Parts: Extracts structured data from PDFs with high
                accuracy
              </li>
              <li>
                Fire Craw: Converts web content into clean, parseable markdown
              </li>
            </ul>
          </div>

          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h4 className="font-semibold">2. Intelligent Chunking</h4>
            <p>
              Optimizing chunk size is crucial for maintaining context while
              fitting within LLM token limits:
            </p>
            <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
              <code>
                {`def adaptive_chunk(text, target_size=500):
  sentences = nltk.sent_tokenize(text)
  chunks = []
  current_chunk = ""
  
  for sentence in sentences:
      if len(current_chunk) + len(sentence) <= target_size:
          current_chunk += sentence + " "
      else:
          chunks.append(current_chunk.strip())
          current_chunk = sentence + " "
  
  if current_chunk:
      chunks.append(current_chunk.strip())
  
  return chunks`}
              </code>
            </pre>
          </div>

          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h4 className="font-semibold">3. Hybrid Search with Reranking</h4>
            <p>
              Combining vector search with keyword matching and using a separate
              model for reranking:
            </p>
            <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
              <code>
                {`def hybrid_search(query, vector_db, keyword_index):
  vector_results = vector_db.similarity_search(query, k=20)
  keyword_results = keyword_index.search(query, k=20)
  
  combined_results = list(set(vector_results + keyword_results))
  
  reranker = SentenceTransformer('cross-encoder/ms-marco-MiniLM-L-6-v2')
  reranke
d_results = reranker.predict([(query, doc.page_content) for doc in combined_results])
  
  return [doc for _, doc in sorted(zip(reranked_results, combined_results), reverse=True)][:10]`}
              </code>
            </pre>
          </div>
        </div>
      </section>

      <section id="pipeline" className="space-y-4">
        <h2 className="text-sm font-semibold">The Agentic RAG Pipeline</h2>
        <p>
          The heart of Local Agentic RAG lies in its intelligent, agent-driven
          pipeline:
        </p>

        <div className="space-y-4">
          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h4 className="font-semibold">1. Query Translation Agent</h4>
            <p>
              This agent reformulates user queries to optimize for retrieval:
            </p>
            <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
              <code>
                {`def query_translator(user_query):
  prompt = f"""
  Translate the following user query into a more comprehensive search query:
  User Query: {user_query}
  Translated Query:
  """
  response = llm(prompt)
  return response.strip()`}
              </code>
            </pre>
          </div>

          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h4 className="font-semibold">2. Metadata Filtering Agent</h4>
            <p>This agent uses metadata to improve search relevance:</p>
            <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
              <code>
                {`def metadata_filter(query, docs):
  prompt = f"""
  Given the query "{query}", filter and rank the following documents based on their metadata relevance:
  {docs}
  Return the indices of the top 5 most relevant documents.
  """
  response = llm(prompt)
  return [int(idx) for idx in response.split()]`}
              </code>
            </pre>
          </div>

          <div className="border border-gray-800 rounded p-3 hover:bg-gray-900 transition-colors">
            <h4 className="font-semibold">3. Corrective RAG Agent</h4>
            <p>This agent ensures high-quality, relevant responses:</p>
            <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
              <code>
                {`def corrective_rag_agent(query, vector_db):
  max_iterations = 3
  for i in range(max_iterations):
      translated_query = query_translator(query)
      relevant_docs = hybrid_search(translated_query, vector_db, keyword_index)
      filtered_docs = [relevant_docs[i] for i in metadata_filter(query, relevant_docs)]
      
      answer = generate_answer(query, filtered_docs)
      
      if not is_hallucinating(answer) and answers_question(answer, query):
          return answer
      
      if i < max_iterations - 1:
          query = refine_query(query, answer)
  
  return "I'm sorry, but I couldn't find a satisfactory answer to your question."

def refine_query(original_query, previous_answer):
  prompt = f"""
  The original query was: "{original_query}"
  The previous answer was: "{previous_answer}"
  This answer was not satisfactory. Please refine the original query to get a better answer.
  Refined query:
  """
  return llm(prompt).strip()`}
              </code>
            </pre>
          </div>
        </div>
      </section>

      <section id="implementation" className="space-y-4">
        <h2 className="text-sm font-semibold">
          Implementing Local Agentic RAG
        </h2>
        <p>To implement Local Agentic RAG, follow these steps:</p>

        <ol className="list-decimal list-inside space-y-2">
          <li>
            <strong>Set up your environment:</strong>
            <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
              <code>
                {`pip install langchain gpt4all sentence-transformers nltk
pip install llama-cpp-python`}
              </code>
            </pre>
          </li>
          <li>
            <strong>Initialize Llama 3:</strong>
            <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
              <code>
                {`from llama_cpp import Llama

llm = Llama(model_path="path/to/llama-3-model.bin", n_ctx=2048, n_threads=4)

def llm(prompt):
  return llm(prompt, max_tokens=100)['choices'][0]['text']`}
              </code>
            </pre>
          </li>
          <li>
            <strong>Set up your vector database and keyword index:</strong>
            <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
              <code>
                {`from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

docs = text_splitter.split_documents(your_documents)
vector_db = Chroma.from_documents(docs, embeddings)

# Set up a simple keyword index (you might want to use a more sophisticated solution in production)
keyword_index = {doc.page_content: doc for doc in docs}`}
              </code>
            </pre>
          </li>
          <li>
            <strong>Implement the Agentic RAG pipeline:</strong>
            <p>
              Use the code snippets provided in the previous section to
              implement the query translator, metadata filter, and corrective
              RAG agent.
            </p>
          </li>
          <li>
            <strong>Use the Local Agentic RAG system:</strong>
            <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
              <code>
                {`user_query = "What are the key benefits of using Local Agentic RAG?"
answer = corrective_rag_agent(user_query, vector_db)
print(answer)`}
              </code>
            </pre>
          </li>
        </ol>
      </section>

      <section id="conclusion" className="space-y-4">
        <h2 className="text-sm font-semibold">
          Conclusion: The Future of LLMs and Private Knowledge
        </h2>
        <p>
          Local Agentic RAG represents a monumental leap forward in our ability
          to leverage LLMs with private, organization-specific knowledge. By
          combining advanced parsing, intelligent chunking, hybrid search, and
          AI agents, we can create a system that truly performs "10x better"
          with private knowledge.
        </p>
        <p>The benefits of this approach are manifold:</p>
        <ul className="list-disc list-inside space-y-2">
          <li>Dramatically improved accuracy and relevance of responses</li>
          <li>Ability to handle complex, multi-faceted queries</li>
          <li>Dynamic adaptation to user intent and context</li>
          <li>Efficient processing of diverse data types</li>
          <li>Reduced hallucination and increased reliability</li>
        </ul>
        <p>
          As we continue to refine and expand upon the Local Agentic RAG
          approach, we open up new possibilities for AI-driven information
          retrieval, knowledge management, and decision support across a wide
          range of industries and applications.
        </p>
        <p>
          The future of LLMs is not just about bigger models or more data—it's
          about smarter, more adaptive systems that can truly understand and
          leverage the unique knowledge within each organization. Local Agentic
          RAG is a significant step towards that future.
        </p>
      </section>

      <section id="video-resources" className="space-y-4">
        <h2 className="text-sm font-semibold">Video Resources</h2>
        <p className="mb-4">
          For a deeper dive into the concepts of Local Agentic RAG, check out
          this informative video:
        </p>
        <a
          href="https://youtu.be/u5Vcrwpzoz8?si=NPD8KV3twNAFDi9g"
          target="_blank"
          rel="noopener noreferrer"
          className="block bg-gray-900 border border-gray-700 rounded-lg overflow-hidden hover:bg-gray-800 transition-colors duration-300"
        >
          <div className="p-4 flex items-center space-x-4">
            <div className="flex-shrink-0">
              <PlayCircle size={48} className="text-gray-400" />
            </div>
            <div className="flex-grow">
              <h3 className="text-sm font-semibold text-gray-200">
                Local Agentic RAG: In-Depth Explanation
              </h3>
              <p className="text-xs text-gray-400">
                Learn about the revolutionary approach to enhancing LLMs with
                private knowledge
              </p>
            </div>
          </div>
        </a>
      </section>

      {showScrollTop && (
        <button
          onClick={scrollToTop}
          className="fixed bottom-4 right-4 bg-gray-800 p-2 rounded-full hover:bg-gray-700 transition-colors"
          aria-label="Scroll to top"
        >
          <ChevronUp size={20} />
        </button>
      )}
    </article>
  );
}
