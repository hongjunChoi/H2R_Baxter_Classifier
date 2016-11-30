public class RoomStack<T> {
	private AtomicInteger top; 
	private int pushRoom;
	private int popRoom; 
	private Room room; 
	public RoomStack(int capacity){
		top = new AtomicInteger();
		pushRoom = 0;
		popRoom = 0;
		room = new Room();
	}

	public void push(T x) throws FullException {
		room.enter(pushRoom);
		
	}

	public T pop() throws EmptyException {
		room.enter(pushRoom);
		room.enter(popRoom);
		
	} 

	//when poping, if 
}