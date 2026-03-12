import { useState } from "react";
import { Upload, FileText, Trash2, CheckCircle, AlertCircle, Menu, MessageSquare, Loader2 } from "lucide-react";
import { Link } from "react-router";

interface Document {
  id: string;
  name: string;
  size: number;
  uploadedAt: Date;
  status: "processing" | "ready" | "error";
  // Stage indicates the current ingestion sub-step shown in UI while processing
  stage?: "parsing" | "chunking" | "embedding" | "storing" | "done" | "idle";
  chunks?: number;
}

export function DocumentUpload() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(1) + " MB";
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      handleFiles(files);
    }
  };

  const handleFiles = async (files: File[]) => {
    const API_BASE = "http://localhost:8000";

    // Add documents to UI with processing status
    const newDocs: Document[] = files.map((file) => ({
      id: Date.now().toString() + Math.random(),
      name: file.name,
      size: file.size,
      uploadedAt: new Date(),
      status: "processing" as const,
      stage: "parsing",
    }));

    setDocuments((prev) => [...newDocs, ...prev]);

    // Upload each file to backend for real ingestion
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const tempDoc = newDocs[i];

      // Upload each file and then subscribe to server-sent events for real-time status

      try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${API_BASE}/upload`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Upload failed: ${response.statusText}`);
        }

        const result = await response.json();
        // Derive a stable doc id from the server response.
        // Server returns index_path like ".../document_store/<base>_pageindex.json"
        let serverDocId = "";
        if (result && result.index_path) {
          const parts = result.index_path.split("/").pop()?.split("\\").pop() || "";
          serverDocId = parts.replace("_pageindex.json", "") || parts;
        } else if (result && result.doc_name) {
          serverDocId = result.doc_name.replace(/\.pdf$/i, "");
        } else {
          // fallback to temp id (shouldn't happen)
          serverDocId = tempDoc.id;
        }

        // Update the temporary doc entry with the server doc_id and current status
        setDocuments((prev) =>
          prev.map((d) =>
            d.id === tempDoc.id
              ? {
                  ...d,
                  id: serverDocId,
                  status: result.index_path ? "ready" : "processing",
                  chunks: result.top_level_structure ? result.top_level_structure.length : d.chunks,
                }
              : d
          )
        );

        // Subscribe to server-sent events for real-time status updates (doc might already be ready)
        try {
          const es = new EventSource(`${API_BASE}/status/stream/${encodeURIComponent(serverDocId)}`);
          es.onmessage = (e) => {
            try {
              const payload = JSON.parse(e.data);
              setDocuments((prev) =>
                prev.map((d) =>
                  d.id === serverDocId
                    ? {
                        ...d,
                        status: payload.status ?? d.status,
                        stage: payload.stage ?? d.stage,
                        chunks: payload.chunks ?? d.chunks,
                      }
                    : d
                )
              );
              if (payload.status === "ready" || payload.status === "error") {
                es.close();
              }
            } catch (err) {
              console.error("Failed to parse SSE payload", err);
            }
          };
          es.onerror = (err) => {
            // Log and close on error; frontend can fallback to GET /status/{id}
            console.error("SSE error", err);
            es.close();
          };
        } catch (err) {
          console.error("Failed to open SSE", err);
        }
      } catch (error) {
        console.error(`Error uploading ${file.name}:`, error);
        setDocuments((prev) =>
          prev.map((d) => (d.id === tempDoc.id ? { ...d, status: "error" as const } : d))
        );
      }
    }
  };

  const handleDelete = async (id: string) => {
    const API_BASE = "http://localhost:8000";

    try {
      const response = await fetch(`${API_BASE}/documents/${id}`, {
        method: "DELETE",
      });

      if (response.ok) {
        setDocuments((prev) => prev.filter((d) => d.id !== id));
      } else {
        console.error("Failed to delete document from backend");
        setDocuments((prev) => prev.filter((d) => d.id !== id));
      }
    } catch (error) {
      console.error("Error deleting document:", error);
      setDocuments((prev) => prev.filter((d) => d.id !== id));
    }
  };

  const getStatusIcon = (status: Document["status"]) => {
    switch (status) {
      case "ready":
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case "processing":
        return <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />;
      case "error":
        return <AlertCircle className="w-4 h-4 text-red-600" />;
    }
  };

  const getStatusText = (status: Document["status"]) => {
    switch (status) {
      case "ready":
        return "Ready";
      case "processing":
        return "Processing";
      case "error":
        return "Error";
    }
  };

  const getStatusColor = (status: Document["status"]) => {
    switch (status) {
      case "ready":
        return "text-green-600";
      case "processing":
        return "text-blue-600";
      case "error":
        return "text-red-600";
    }
  };

  // Real-time status will be updated from server via SSE; no local simulation needed.

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button className="lg:hidden p-2 hover:bg-gray-100 rounded-lg">
              <Menu className="w-5 h-5 text-gray-600" />
            </button>
            <div className="flex items-center gap-3">
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                <rect width="32" height="32" rx="6" fill="#1a73e8"/>
                <path d="M16 8L8 13v10l8 5 8-5V13l-8-5z" fill="white" opacity="0.9"/>
                <circle cx="16" cy="16" r="3" fill="white"/>
              </svg>
              <div>
                <h1 className="text-xl text-gray-900">DocuMind</h1>
              </div>
            </div>
          </div>
          <Link
            to="/"
            className="px-4 py-2 text-sm font-medium text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
          >
            Go to chat
          </Link>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Page Title */}
        <div className="mb-8">
          <h2 className="text-2xl text-gray-900 mb-2">Document management</h2>
          <p className="text-sm text-gray-600">
            Upload and manage company documents for the AI knowledge base
          </p>
        </div>

        

        {/* Upload Area */}
        <div className="bg-white rounded-lg border border-gray-200 p-8 mb-8">
          <h3 className="text-lg text-gray-900 mb-4">Upload documents</h3>
          
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`relative border-2 border-dashed rounded-lg p-12 transition-all ${
              isDragging
                ? "border-blue-600 bg-blue-50"
                : "border-gray-300 hover:border-gray-400"
            }`}
          >
            <input
              type="file"
              multiple
              accept=".pdf,.docx,.txt,.md"
              onChange={handleFileInput}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <div className="flex flex-col items-center text-center">
              <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                <Upload className="w-6 h-6 text-gray-600" />
              </div>
              <h3 className="text-base text-gray-900 mb-1">
                {isDragging ? "Drop files here" : "Drag files here or click to browse"}
              </h3>
              <p className="text-sm text-gray-600">
                Supported formats: PDF, DOCX, TXT, MD (up to 50MB each)
              </p>
            </div>
          </div>
        </div>

        {/* Documents List */}
        <div className="bg-white rounded-lg border border-gray-200">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg text-gray-900">Documents</h3>
          </div>

          {documents.length > 0 ? (
            <div className="divide-y divide-gray-200">
              {documents.map((doc) => (
                <div
                  key={doc.id}
                  className="px-6 py-4 hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded bg-blue-50 flex items-center justify-center flex-shrink-0">
                      <FileText className="w-5 h-5 text-blue-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4 className="text-sm font-medium text-gray-900 truncate">{doc.name}</h4>
                      <div className="flex items-center gap-3 mt-1 text-xs text-gray-600">
                        <span>{formatFileSize(doc.size)}</span>
                        <span>•</span>
                        <span>Uploaded {doc.uploadedAt.toLocaleDateString()}</span>
                        {doc.chunks && (
                          <>
                            <span>•</span>
                            <span>{doc.chunks} chunks</span>
                          </>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className={`flex items-center gap-2 text-sm ${getStatusColor(doc.status)}`}>
                        {getStatusIcon(doc.status)}
                        <span>
                          {doc.status === "processing"
                            ? doc.stage
                              ? `${doc.stage.charAt(0).toUpperCase() + doc.stage.slice(1)}...`
                              : "Processing"
                            : getStatusText(doc.status)}
                        </span>
                      </div>
                      <button
                        onClick={() => handleDelete(doc.id)}
                        className="p-2 hover:bg-gray-100 rounded-lg text-gray-600 hover:text-red-600 transition-colors"
                        title="Delete document"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="px-6 py-12 text-center">
              <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center mx-auto mb-4">
                <FileText className="w-6 h-6 text-gray-400" />
              </div>
              <p className="text-sm text-gray-600">No documents uploaded</p>
              <p className="text-xs text-gray-500 mt-1">Upload your first document to get started</p>
            </div>
          )}
        </div>

        
      </div>
    </div>
  );
}
