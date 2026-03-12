import { useState, useRef, useEffect } from "react";
import { Send, Bot, FileText, X } from "lucide-react";
import { Link } from "react-router";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  sources?: Source[];
  isStreaming?: boolean;
}

interface Source {
  name: string;
  page?: number;
  excerpt: string;
}

interface SourcePreview {
  source: Source;
  fullContext: string;
  highlightedText: string;
}

interface QueryResponse {
  answer: string;
  sources: string[];
  chunks: Array<{
    text: string;
    filename: string;
    chunk_index: number;
    relevance_score: number;
  }>;
  conversation_id: string;
}

const API_BASE = "http://localhost:8000";

export function ChatInterface() {
  const [reasoningTrace, setReasoningTrace] = useState<string[]>([]);
  const [showReasoning, setShowReasoning] = useState<boolean>(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content: "Hello! I'm your AI assistant trained on your uploaded documents. How can I help you today?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sourcePreview, setSourcePreview] = useState<SourcePreview | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [documentCount, setDocumentCount] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const streamingTimeoutRef = useRef<NodeJS.Timeout>();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    return () => {
      if (streamingTimeoutRef.current) {
        clearTimeout(streamingTimeoutRef.current);
      }
    };
  }, []);

  const streamText = (fullText: string, messageId: string, sources: Source[]) => {
    // For faster responses, stream by words instead of characters
    const words = fullText.split(" ");
    let currentIndex = 0;
    const streamInterval = 30; // milliseconds between each word chunk

    const stream = () => {
      if (currentIndex < words.length) {
        const chunk = words.slice(0, currentIndex + 1).join(" ");
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId
              ? { ...msg, content: chunk, isStreaming: true }
              : msg
          )
        );
        currentIndex++;
        streamingTimeoutRef.current = setTimeout(stream, streamInterval);
      } else {
        // Streaming complete
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId ? { ...msg, isStreaming: false, sources } : msg
          )
        );
        setIsLoading(false);
      }
    };

    stream();
  };

  // Fetch document count on mount
  useEffect(() => {
    const fetchDocCount = async () => {
      try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
          const data = await response.json();
          setDocumentCount(data.documents_indexed || 0);
        }
      } catch (err) {
        console.error("Failed to fetch document count:", err);
      }
    };
    fetchDocCount();
  }, []);

  const handleSend = async (question?: string) => {
    const messageText = question || input.trim();
    if (!messageText || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: messageText,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Create empty assistant message for streaming effect
    const assistantMessageId = (Date.now() + 1).toString();
    const emptyAssistantMessage: Message = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      timestamp: new Date(),
      isStreaming: true,
    };

    setMessages((prev) => [...prev, emptyAssistantMessage]);

    try {
      // Streamed call to backend
      const response = await fetch(`${API_BASE}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: messageText,
          conversation_id: conversationId || undefined,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`Query failed: ${response.statusText}`);
      }

      // Show reasoning panel
      setShowReasoning(true);
      setReasoningTrace([]);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      // Helper to append assistant content
      const appendAssistant = (text: string) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId ? { ...msg, content: (msg.content || "") + text } : msg
          )
        );
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Split SSE-style payloads (separated by double newline)
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          const line = part.trim();
          if (!line) continue;
          if (!line.startsWith("data:")) continue;
          const jsonText = line.replace(/^data:\s*/, "");
          let ev;
          try {
            ev = JSON.parse(jsonText);
          } catch (err) {
            console.error("Failed to parse event JSON", err, jsonText);
            continue;
          }

          if (ev.type === "trace") {
            setReasoningTrace((prev) => [...prev, ev.payload]);
          } else if (ev.type === "final_trace") {
            // payload is array of trace lines
            setReasoningTrace((prev) => [...prev, ...(ev.payload || [])]);
          } else if (ev.type === "answer_chunk") {
            appendAssistant(ev.payload);
          } else if (ev.type === "done") {
            // final metadata available in ev.payload
            // finalize assistant message (no follow-up suggestions)
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessageId ? { ...msg, isStreaming: false } : msg
              )
            );
            // store conversation id if returned
            if (ev.payload && ev.payload.conversation_id) {
              setConversationId(ev.payload.conversation_id);
            }
            setIsLoading(false);
          }
        }
      }
    } catch (error) {
      console.error("Query error:", error);
      
      // Update message with error
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: "I apologize, but I encountered an error while processing your question. Please make sure documents are uploaded and the backend is running.",
                isStreaming: false,
              }
            : msg
        )
      );
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };
 

  const handleSourceClick = (source: Source) => {
    // Show source with context - in production this could fetch full document
    const fullContext = source.excerpt;
    
    setSourcePreview({ 
      source, 
      fullContext: fullContext,
      highlightedText: source.excerpt,
    });
  };

  const handleClearConversation = async () => {
    if (conversationId) {
      try {
        await fetch(`${API_BASE}/chat/${conversationId}`, { method: "DELETE" });
      } catch (err) {
        console.error("Failed to clear conversation:", err);
      }
    }
    setConversationId(null);
    setMessages([
      {
        id: "1",
        role: "assistant",
        content: "Hello! I'm your AI assistant trained on your uploaded documents. How can I help you today?",
        timestamp: new Date(),
      },
    ]);
  };

  const closeSourcePreview = () => {
    setSourcePreview(null);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-6">
      <div className="w-full max-w-4xl h-[90vh] bg-white rounded-2xl shadow-xl overflow-hidden flex flex-col">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-4 border-b">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-indigo-600 to-blue-400 flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-gray-900">DocuMind</h1>
              <div className="mt-1 text-xs text-gray-500">🧠 Reasoning Over Retrieval · 🚫 No Vector DB · 📑 Hierarchical · 🔍 Human-like</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {messages.length > 1 && (
              <button
                onClick={handleClearConversation}
                className="px-3 py-1 text-sm font-medium text-gray-600 hover:bg-gray-100 rounded"
                title="Clear conversation history"
              >
                New chat
              </button>
            )}
            <Link to="/upload" className="px-3 py-1 text-sm font-medium text-blue-600 hover:bg-blue-50 rounded">
              Manage documents
            </Link>
          </div>
        </header>

        {/* Messages */}
        <main className="flex-1 overflow-auto bg-gray-50 p-6">
          <div className="flex flex-col gap-4 max-w-3xl mx-auto">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {message.role === "assistant" && (
                  <div className="flex-shrink-0 mr-3">
                    <div className="w-9 h-9 rounded-full bg-gray-100 flex items-center justify-center">
                      <Bot className="w-4 h-4 text-gray-700" />
                    </div>
                  </div>
                )}

                <div className={`max-w-[70%]`}>
                  <div
                    className={`px-4 py-3 rounded-2xl shadow-sm ${
                      message.role === "user"
                        ? "bg-indigo-600 text-white rounded-br-xl"
                        : "bg-white border border-gray-200 text-gray-900 rounded-bl-xl"
                    }`}
                  >
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                  </div>
                  {message.sources && message.sources.length > 0 && !message.isStreaming && (
                    <div className="mt-2 flex flex-wrap gap-2">
                      {message.sources.map((source, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleSourceClick(source)}
                          className="text-xs bg-white px-2 py-1 rounded border text-gray-700"
                        >
                          {source.name}{source.page ? ` • p.${source.page}` : ""}
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {message.role === "user" && (
                  <div className="flex-shrink-0 ml-3">
                    <div className="w-9 h-9 rounded-full bg-indigo-600 flex items-center justify-center text-white">
                      You
                    </div>
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </main>

        {/* Input */}
        <footer className="px-6 py-4 border-t bg-white">
          <div className="max-w-4xl mx-auto flex items-center gap-3">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about your documents..."
              className="flex-1 px-4 py-3 rounded-xl border border-gray-200 focus:outline-none focus:ring-1 focus:ring-indigo-300 resize-none"
              rows={1}
              style={{ minHeight: "48px", maxHeight: "160px" }}
            />
            <button
              onClick={() => handleSend()}
              disabled={!input.trim() || isLoading}
              className="w-12 h-12 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-300 flex items-center justify-center text-white"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </footer>
      </div>

      {/* Source Preview Modal */}
      {sourcePreview && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-3xl w-full max-h-[80vh] flex flex-col overflow-hidden">
            <div className="flex items-center justify-between px-6 py-4 border-b">
              <div>
                <h2 className="text-lg font-medium text-gray-900">Source Preview</h2>
                <p className="text-sm text-gray-600 mt-0.5">
                  {sourcePreview.source.name} {sourcePreview.source.page && `• Page ${sourcePreview.source.page}`}
                </p>
              </div>
              <button className="p-2 hover:bg-gray-100 rounded" onClick={closeSourcePreview}>
                <X className="w-5 h-5 text-gray-600" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto px-6 py-4">
              <div className="space-y-4">
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <div className="w-1 h-4 bg-indigo-600 rounded"></div>
                    <h3 className="text-sm font-medium text-gray-900">Referenced excerpt</h3>
                  </div>
                  <div className="bg-indigo-50 border-l-4 border-indigo-600 px-4 py-3 rounded-r">
                    <p className="text-sm text-gray-900 leading-relaxed">{sourcePreview.highlightedText}</p>
                  </div>
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <div className="w-1 h-4 bg-gray-400 rounded"></div>
                    <h3 className="text-sm font-medium text-gray-900">Full context</h3>
                  </div>
                  <div className="bg-gray-50 px-4 py-3 rounded border border-gray-200">
                    <p className="text-sm text-gray-700 leading-relaxed">
                      {sourcePreview.fullContext.split(sourcePreview.highlightedText).map((part, idx, arr) => (
                        <span key={idx}>
                          {part}
                          {idx < arr.length - 1 && (
                            <mark className="bg-yellow-200 px-1 rounded">{sourcePreview.highlightedText}</mark>
                          )}
                        </span>
                      ))}
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div className="px-6 py-4 border-t bg-gray-50">
              <p className="text-xs text-gray-600">This preview shows the exact passage from the document that was used to generate the AI response.</p>
            </div>
          </div>
        </div>
      )}

      {/* Source Preview Modal */}
      {sourcePreview && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[80vh] flex flex-col">
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
              <div>
                <h2 className="text-lg font-medium text-gray-900">Source Preview</h2>
                <p className="text-sm text-gray-600 mt-0.5">
                  {sourcePreview.source.name} {sourcePreview.source.page && `• Page ${sourcePreview.source.page}`}
                </p>
              </div>
              <button 
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors" 
                onClick={closeSourcePreview}
              >
                <X className="w-5 h-5 text-gray-600" />
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto px-6 py-4">
              <div className="space-y-4">
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <div className="w-1 h-4 bg-blue-600 rounded"></div>
                    <h3 className="text-sm font-medium text-gray-900">Referenced excerpt</h3>
                  </div>
                  <div className="bg-blue-50 border-l-4 border-blue-600 px-4 py-3 rounded-r">
                    <p className="text-sm text-gray-900 leading-relaxed">
                      {sourcePreview.highlightedText}
                    </p>
                  </div>
                </div>

                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <div className="w-1 h-4 bg-gray-400 rounded"></div>
                    <h3 className="text-sm font-medium text-gray-900">Full context</h3>
                  </div>
                  <div className="bg-gray-50 px-4 py-3 rounded border border-gray-200">
                    <p className="text-sm text-gray-700 leading-relaxed">
                      {sourcePreview.fullContext.split(sourcePreview.highlightedText).map((part, idx, arr) => (
                        <span key={idx}>
                          {part}
                          {idx < arr.length - 1 && (
                            <mark className="bg-yellow-200 px-1 rounded">
                              {sourcePreview.highlightedText}
                            </mark>
                          )}
                        </span>
                      ))}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
              <p className="text-xs text-gray-600">
                This preview shows the exact passage from the document that was used to generate the AI response.
              </p>
            </div>
          </div>
        </div>
      )}
      {/* Reasoning Sidebar */}
      {showReasoning && (
        <div className="fixed right-4 top-16 w-80 max-h-[70vh] bg-white border border-gray-200 shadow-lg rounded-lg p-4 z-50 overflow-auto">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-sm font-medium text-gray-900">Reasoning Trace</h4>
            <button
              onClick={() => setShowReasoning(false)}
              className="p-1 hover:bg-gray-100 rounded"
            >
              <X className="w-4 h-4 text-gray-600" />
            </button>
          </div>
          <div className="text-xs text-gray-600 space-y-2">
            {reasoningTrace.length === 0 ? (
              <div className="text-center text-gray-400">Waiting for trace...</div>
            ) : (
              reasoningTrace.map((line, idx) => (
                <div key={idx} className="p-2 bg-gray-50 rounded border border-gray-100">
                  <pre className="whitespace-pre-wrap text-xs">{line}</pre>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
