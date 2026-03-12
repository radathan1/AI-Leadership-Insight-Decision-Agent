import { BrowserRouter, Routes, Route } from "react-router";
import { ChatInterface } from "./components/ChatInterface";
import { DocumentUpload } from "./components/DocumentUpload";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ChatInterface />} />
        <Route path="/upload" element={<DocumentUpload />} />
      </Routes>
    </BrowserRouter>
  );
}
