import React, { useState, useEffect, useRef } from 'react';
import { Icons } from './Icon';

export const Terminal: React.FC = () => {
  const [logs, setLogs] = useState<string[]>([
    'flexcyz@dev:~$ docker-compose up -d',
    'Starting flexcyz_redis_1 ... done',
    'Starting flexcyz_backend_1 ... done',
    'Attaching to flexcyz_backend_1',
    'backend_1  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)',
    'backend_1  | INFO:     Started server process [1]',
    'backend_1  | INFO:     Waiting for application startup.',
    'backend_1  | INFO:     Application startup complete.',
    'flexcyz@dev:~$ '
  ]);
  
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="h-48 bg-[#0f0f14] border-t border-gray-800 flex flex-col text-sm font-mono">
      <div className="flex items-center justify-between px-4 py-1 bg-editor-sidebar border-b border-gray-800">
        <div className="flex items-center gap-4">
           <div className="flex items-center gap-1 text-orchid-400 border-b border-orchid-500 pb-0.5">
             <Icons.Terminal size={12} />
             <span className="text-xs font-bold uppercase">Terminal</span>
           </div>
           <div className="text-gray-500 text-xs hover:text-gray-300 cursor-pointer">Output</div>
           <div className="text-gray-500 text-xs hover:text-gray-300 cursor-pointer">Problems</div>
        </div>
        <div className="flex gap-2">
            <Icons.ChevronDown size={14} className="text-gray-500 cursor-pointer" />
            <Icons.Close size={14} className="text-gray-500 cursor-pointer" />
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-2 text-gray-400">
        {logs.map((log, i) => (
          <div key={i} className="mb-0.5">
            <span className={log.startsWith('flexcyz') ? 'text-green-400' : ''}>
                {log}
            </span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
};