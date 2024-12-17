import React, { useEffect, useState } from 'react';

export default function TrainStatus({ trainingStarted }: { trainingStarted: boolean }) {
    const [status, setStatus] = useState('idle');

    useEffect(() => {
        if (trainingStarted) {
            const eventSource = new EventSource('http://127.0.0.1:5000/train-status');

            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.status === 'running') {
                    setStatus('running');
                } else if (data.status === 'stopped') {
                    setStatus('stopped');
                    eventSource.close();
                }
            };

            eventSource.onerror = (error) => {
                console.error("SSE connection error:", error);
                eventSource.close();
                setTimeout(() => {
                    const newEventSource = new EventSource('http://127.0.0.1:5000/train-status');
                    newEventSource.onmessage = eventSource.onmessage;
                    newEventSource.onerror = eventSource.onerror;
                }, 5000);
            };

            return () => {
                eventSource.close();
            };
        }
    }, [trainingStarted]);

    return (
        <div>
            <h1>Training Status: {status}</h1>
            {status === 'idle' && <p>No training in progress.</p>}
            {status === 'running' && <p>Training in progress...</p>}
            {status === 'stopped' && <p>Training completed ~</p>}
        </div>
    );
}