import React from 'react';
import { 
  Folder, 
  FolderOpen, 
  File, 
  FileCode, 
  FileJson, 
  ChevronRight, 
  ChevronDown, 
  Terminal, 
  Play, 
  Search,
  Settings,
  Menu,
  X
} from 'lucide-react';

export const Icons = {
  Folder,
  FolderOpen,
  File,
  FileCode,
  FileJson,
  ChevronRight,
  ChevronDown,
  Terminal,
  Play,
  Search,
  Settings,
  Menu,
  Close: X
};

export const FileIcon: React.FC<{ name: string; className?: string }> = ({ name, className }) => {
  if (name.endsWith('.py')) return <Icons.FileCode className={`text-blue-400 ${className}`} />;
  if (name.endsWith('.js') || name.endsWith('.ts') || name.endsWith('.tsx')) return <Icons.FileCode className={`text-yellow-400 ${className}`} />;
  if (name.endsWith('.html')) return <Icons.FileCode className={`text-orange-400 ${className}`} />;
  if (name.endsWith('.css')) return <Icons.FileCode className={`text-blue-300 ${className}`} />;
  if (name.endsWith('.json') || name.endsWith('.yaml') || name.endsWith('.yml')) return <Icons.FileJson className={`text-green-400 ${className}`} />;
  return <Icons.File className={`text-gray-400 ${className}`} />;
};