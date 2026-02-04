import React from 'react';
import { Tab, FileNode } from '../types';
import { Icons } from './Icon';

interface EditorProps {
  activeTabId: string | null;
  tabs: Tab[];
  onCloseTab: (id: string) => void;
  onSelectTab: (id: string) => void;
  onContentChange: (id: string, newContent: string) => void;
}

export const Editor: React.FC<EditorProps> = ({
  activeTabId,
  tabs,
  onCloseTab,
  onSelectTab,
  onContentChange,
}) => {
  const activeTab = tabs.find((t) => t.id === activeTabId);

  if (!activeTab) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-gray-500 bg-editor-bg">
        <Icons.Terminal size={64} className="mb-4 text-orchid-900 opacity-50" />
        <p className="text-lg">Select a file to start editing</p>
        <p className="text-sm opacity-60 mt-2">Flexcyz v1.0.0 Dev Build</p>
      </div>
    );
  }

  // Simple line numbers calculation
  const lines = activeTab.content.split('\n');

  return (
    <div className="h-full flex flex-col bg-editor-bg">
      {/* Tabs Bar */}
      <div className="flex bg-editor-sidebar overflow-x-auto border-b border-gray-800">
        {tabs.map((tab) => (
          <div
            key={tab.id}
            className={`group flex items-center min-w-[120px] max-w-[200px] px-3 py-2 text-sm border-r border-gray-800 cursor-pointer select-none ${
              tab.id === activeTabId
                ? 'bg-editor-bg text-orchid-100 border-t-2 border-t-orchid-500'
                : 'bg-editor-sidebar text-gray-500 hover:bg-[#1e1e2e] hover:text-gray-300'
            }`}
            onClick={() => onSelectTab(tab.id)}
          >
            <span className="flex-1 truncate mr-2">{tab.name}</span>
            <button
              className={`opacity-0 group-hover:opacity-100 hover:bg-gray-700 rounded p-0.5 transition-opacity ${
                tab.isDirty ? 'opacity-100' : ''
              }`}
              onClick={(e) => {
                e.stopPropagation();
                onCloseTab(tab.id);
              }}
            >
              {tab.isDirty ? (
                 <div className="w-2 h-2 rounded-full bg-orchid-400 mx-0.5" />
              ) : (
                <Icons.Close size={12} />
              )}
            </button>
          </div>
        ))}
      </div>

      {/* Editor Content Area */}
      <div className="flex-1 overflow-hidden flex relative font-mono text-sm">
        {/* Line Numbers */}
        <div className="w-12 bg-editor-bg border-r border-gray-800 text-right pr-3 pt-4 text-gray-600 select-none overflow-hidden flex-shrink-0">
          {lines.map((_, i) => (
            <div key={i} className="leading-6">
              {i + 1}
            </div>
          ))}
        </div>

        {/* Text Area */}
        <textarea
          className="flex-1 bg-editor-bg text-editor-text p-4 pt-4 outline-none resize-none leading-6 w-full h-full whitespace-pre font-mono"
          value={activeTab.content}
          onChange={(e) => onContentChange(activeTab.id, e.target.value)}
          spellCheck={false}
        />
      </div>
    </div>
  );
};