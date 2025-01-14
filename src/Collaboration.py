import socketio
import asyncio
import json

class CollaborationManager:
    def __init__(self):
        self.sio = socketio.AsyncClient()
        self.room = None

    async def connect(self, url):
        await self.sio.connect(url)

    async def join_room(self, room):
        self.room = room
        await self.sio.emit('join', {'room': room})

    async def leave_room(self):
        if self.room:
            await self.sio.emit('leave', {'room': self.room})
            self.room = None

    async def send_update(self, update):
        if self.room:
            await self.sio.emit('update', {'room': self.room, 'data': update})

    async def start(self):
        @self.sio.on('update')
        async def on_update(data):
            # Handle incoming updates
            print(f"Received update: {data}")

        await self.sio.wait()

# Server-side (using Python with aiohttp and socket.io)
import socketio

sio = socketio.AsyncServer(async_mode='aiohttp')
app = web.Application()
sio.attach(app)

@sio.on('join')
async def join(sid, data):
    room = data['room']
    sio.enter_room(sid, room)
    await sio.emit('user_joined', {'user': sid}, room=room)

@sio.on('leave')
async def leave(sid, data):
    room = data['room']
    sio.leave_room(sid, room)
    await sio.emit('user_left', {'user': sid}, room=room)

@sio.on('update')
async def update(sid, data):
    room = data['room']
    await sio.emit('update', data['data'], room=room, skip_sid=sid)


