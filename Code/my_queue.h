#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

typedef long long int LLI;

typedef struct my_queue {
	//𝑄.𝑎𝑟𝑟 stores the elements
	// int* arr;
	MKL_INT* arr;

	//𝑄.𝑐𝑎𝑝𝑎𝑐𝑖𝑡𝑦 stores its maximum size(𝑄.𝑐𝑎𝑝𝑎𝑐𝑖𝑡𝑦 = 𝑄.𝑎𝑟𝑟.𝑙𝑒𝑛𝑔𝑡ℎ)
	LLI capacity;

	//𝑄.𝑓𝑟𝑜𝑛𝑡 stores the front position
	LLI front;

	//𝑄.𝑟𝑒𝑎𝑟 stores the rear position
	LLI rear;
}Queue;

bool isEmpty(Queue* Q)
{
	return Q->front == Q->rear;
}

bool isFull(Queue* Q)
{
	//We leave one element as empty, then we will get full queue for (capacity - 1)
	return Q->front == (Q->rear + 1) % Q->capacity;
}


int NNZ(Queue* Q)
{
	if(isEmpty(Q)){
		return 0;
	}
	if(isFull(Q)){
		return Q->capacity;
	}

	LLI front = Q->front;

	LLI rear = Q->rear;

	if(front < rear){
		return rear - front;
	}
	else{//写了，但是用不到
		return Q->capacity - (front - rear);
	}
}


void enqueue(Queue *Q, int e)
{
	/*if(isFull(Q))
	{
		printf("error Queue full");
		throw "The queue is full, no room to enqueue!";
	}*/
	assert(!isFull(Q));
	Q->arr[Q->rear] = e;
	Q->rear = (Q->rear + 1) % Q->capacity;
}



int dequeue(Queue* Q)
{
	/*if(isEmpty(Q))
	{
		printf("error Queue empty");
		throw "The Queue is empty, no element to deque!";
	}*/
	assert(!isEmpty(Q));
	int e = Q->arr[Q->front];
	Q->front = (Q->front + 1) % Q->capacity;
	return e;
}


int get_front(Queue* Q)
{
	return Q->arr[Q->front];
}

// //Display each element in the Queue
// void display(Queue* Q) {
// 	LLI start = Q->front;
// 	LLI end = Q->rear;
// 	printf("Q->front = %d, Q->rear = %d\n", start, end);
// 	//这里写的不对，要把end和start分开，end可能在start之前，毕竟是循环queue
// 	for (LLI i = start; i != end; i++) 
// 	{
// 		printf("%d->", Q->arr[i]);
// 	}
// 	printf("\nEnd!");
// 	printf("\n");
// }



// int main() {
// 	//We will use capacity-1 to denote the true capacity of the queue
// 	int capacity = 5;

	
// 	//Create Queue
// 	Queue Q =
// 	{
// 		//.arr = 
// 		(int*)malloc(sizeof(int) * capacity),
// 		//.capacity = 
// 		capacity,
// 		//.front =
// 		0,
// 		//.rear = 
// 		0
// 	};


// 	//Display the current queue
// 	display(&Q);

// 	//Insert three elements 1, 2, 3 to the LinkedList
// 	enqueue(&Q, 0.02);
// 	enqueue(&Q, 0.03);
// 	enqueue(&Q, 0.05);
// 	enqueue(&Q, 0.09);

// 	printf("After four enqueue operations: \n");
// 	display(&Q);

// 	printf("The front of queue: %d \n", get_front(&Q));

// 	dequeue(&Q);
// 	dequeue(&Q);

// 	printf("After 2 dequeue operations：\n");
// 	display(&Q);


// 	return 0;
// }