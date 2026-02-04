export type FileType = 'file' | 'folder';

export interface FileNode {
  id: string;
  name: string;
  type: FileType;
  content?: string; // Only for files
  children?: FileNode[]; // Only for folders
  isOpen?: boolean; // UI state for folders
  language?: string; // For syntax highlighting hints
}

export interface Tab {
  id: string;
  fileId: string;
  name: string;
  content: string;
  isDirty: boolean;
}