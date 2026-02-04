import React from 'react';
import { FileNode } from '../types';
import { Icons, FileIcon } from './Icon';

interface SidebarProps {
  files: FileNode[];
  onToggleFolder: (id: string) => void;
  onSelectFile: (file: FileNode) => void;
  selectedFileId: string | null;
}

const FileTreeNode: React.FC<{
  node: FileNode;
  level: number;
  onToggleFolder: (id: string) => void;
  onSelectFile: (file: FileNode) => void;
  selectedFileId: string | null;
}> = ({ node, level, onToggleFolder, onSelectFile, selectedFileId }) => {
  const isSelected = node.id === selectedFileId;
  const paddingLeft = `${level * 1.5}rem`;

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (node.type === 'folder') {
      onToggleFolder(node.id);
    } else {
      onSelectFile(node);
    }
  };

  return (
    <div className="select-none">
      <div
        className={`flex items-center py-1 pr-2 cursor-pointer hover:bg-editor-active transition-colors duration-100 ${
          isSelected ? 'bg-editor-active text-orchid-400' : 'text-gray-400'
        }`}
        style={{ paddingLeft }}
        onClick={handleClick}
      >
        <span className="mr-1.5 opacity-70">
          {node.type === 'folder' ? (
            node.isOpen ? (
              <Icons.ChevronDown size={14} />
            ) : (
              <Icons.ChevronRight size={14} />
            )
          ) : (
            <span className="w-3.5 inline-block" /> 
          )}
        </span>
        
        <span className="mr-2">
            {node.type === 'folder' ? (
               node.isOpen ? <Icons.FolderOpen size={16} className="text-orchid-500" /> : <Icons.Folder size={16} className="text-orchid-500" />
            ) : (
                <FileIcon name={node.name} className="w-4 h-4" />
            )}
        </span>
        
        <span className="text-sm truncate">{node.name}</span>
      </div>

      {node.type === 'folder' && node.isOpen && node.children && (
        <div>
          {node.children.map((child) => (
            <FileTreeNode
              key={child.id}
              node={child}
              level={level + 1}
              onToggleFolder={onToggleFolder}
              onSelectFile={onSelectFile}
              selectedFileId={selectedFileId}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export const Sidebar: React.FC<SidebarProps> = ({ files, onToggleFolder, onSelectFile, selectedFileId }) => {
  return (
    <div className="h-full bg-editor-sidebar border-r border-gray-800 overflow-y-auto">
      <div className="p-3 uppercase text-xs font-bold text-gray-500 tracking-wider flex items-center justify-between">
        <span>Explorer</span>
        <Icons.Settings size={14} className="cursor-pointer hover:text-white" />
      </div>
      <div className="pb-4">
        {files.map((node) => (
          <FileTreeNode
            key={node.id}
            node={node}
            level={1}
            onToggleFolder={onToggleFolder}
            onSelectFile={onSelectFile}
            selectedFileId={selectedFileId}
          />
        ))}
      </div>
    </div>
  );
};