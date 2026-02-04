import React, { useState, useCallback } from 'react';
import { Sidebar } from './components/Sidebar';
import { Editor } from './components/Editor';
import { Terminal } from './components/Terminal';
import { INITIAL_FILE_TREE } from './constants';
import { FileNode, Tab } from './types';
import { Icons } from './components/Icon';

const App: React.FC = () => {
  const [files, setFiles] = useState<FileNode[]>(INITIAL_FILE_TREE);
  const [tabs, setTabs] = useState<Tab[]>([]);
  const [activeTabId, setActiveTabId] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // Recursive function to toggle folder open state
  const toggleFolder = useCallback((id: string) => {
    setFiles((prevFiles) => {
      const updateNodes = (nodes: FileNode[]): FileNode[] => {
        return nodes.map((node) => {
          if (node.id === id) {
            return { ...node, isOpen: !node.isOpen };
          }
          if (node.children) {
            return { ...node, children: updateNodes(node.children) };
          }
          return node;
        });
      };
      return updateNodes(prevFiles);
    });
  }, []);

  const openFile = useCallback((file: FileNode) => {
    if (file.type !== 'file') return;

    setTabs((prevTabs) => {
      const existingTab = prevTabs.find((t) => t.fileId === file.id);
      if (existingTab) {
        setActiveTabId(existingTab.id);
        return prevTabs;
      }
      
      const newTab: Tab = {
        id: crypto.randomUUID(),
        fileId: file.id,
        name: file.name,
        content: file.content || '',
        isDirty: false,
      };
      setActiveTabId(newTab.id);
      return [...prevTabs, newTab];
    });
  }, []);

  const closeTab = useCallback((tabId: string) => {
    setTabs((prev) => {
      const newTabs = prev.filter((t) => t.id !== tabId);
      if (activeTabId === tabId) {
        setActiveTabId(newTabs.length > 0 ? newTabs[newTabs.length - 1].id : null);
      }
      return newTabs;
    });
  }, [activeTabId]);

  const updateTabContent = useCallback((tabId: string, newContent: string) => {
    setTabs((prev) =>
      prev.map((t) =>
        t.id === tabId ? { ...t, content: newContent, isDirty: true } : t
      )
    );
  }, []);

  return (
    <div className="flex h-screen w-screen bg-[#11111b] text-gray-200 overflow-hidden">
      {/* Activity Bar (Leftmost narrow strip) */}
      <div className="w-12 bg-[#181825] border-r border-gray-800 flex flex-col items-center py-4 gap-6">
        <Icons.FileCode 
            size={24} 
            className={`cursor-pointer transition-colors ${isSidebarOpen ? 'text-orchid-500' : 'text-gray-500 hover:text-gray-300'}`} 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
        />
        <Icons.Search size={24} className="text-gray-500 cursor-pointer hover:text-gray-300" />
        <Icons.Play size={24} className="text-gray-500 cursor-pointer hover:text-gray-300" />
        <div className="flex-1" />
        <Icons.Settings size={24} className="text-gray-500 cursor-pointer hover:text-gray-300" />
      </div>

      {/* Sidebar (File Tree) */}
      {isSidebarOpen && (
        <div className="w-64 flex-shrink-0 transition-all duration-300 ease-in-out">
          <Sidebar
            files={files}
            onToggleFolder={toggleFolder}
            onSelectFile={openFile}
            selectedFileId={activeTabId ? tabs.find(t => t.id === activeTabId)?.fileId || null : null}
          />
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Navigation/Breadcrumbs could go here */}
        
        <div className="flex-1 flex flex-col min-h-0">
          <Editor
            activeTabId={activeTabId}
            tabs={tabs}
            onCloseTab={closeTab}
            onSelectTab={setActiveTabId}
            onContentChange={updateTabContent}
          />
          <Terminal />
        </div>
      </div>
      
      {/* Status Bar */}
      <div className="absolute bottom-0 w-full h-6 bg-orchid-900 text-orchid-100 flex items-center px-4 text-xs select-none z-10 justify-between">
        <div className="flex gap-4">
            <span className="flex items-center gap-1"><Icons.Terminal size={10} /> master*</span>
            <span>0 errors, 0 warnings</span>
        </div>
        <div className="flex gap-4">
            <span>Ln 12, Col 34</span>
            <span>UTF-8</span>
            <span>Python</span>
            <span className="hover:bg-orchid-700 px-1 rounded cursor-pointer">Flexcyz Agent System</span>
        </div>
      </div>
    </div>
  );
};

export default App;