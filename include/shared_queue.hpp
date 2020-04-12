#ifndef SHARED_QUEUE_HPP
#define SHARED_QUEUE_HPP

#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T> class SharedQueue
{
public:
	SharedQueue();
	~SharedQueue();

	T& front();
	void pop_front();

	void push_back(const T& item);
	void push_back(T&& item);

	int size();
	unsigned int counted_size();
	bool empty();

private:
	std::deque<T> queue_;
	std::mutex mutex_;
	std::condition_variable cond_;

	unsigned int queue_size_ = 0;
}; 

template <typename T> SharedQueue<T>::SharedQueue() {}

template <typename T> SharedQueue<T>::~SharedQueue() {}

template <typename T> T& SharedQueue<T>::front()
{
	std::unique_lock<std::mutex> mlock(mutex_);

	while (queue_.empty())
		cond_.wait(mlock);
	return queue_.front();
}

template <typename T> void SharedQueue<T>::pop_front()
{
	std::unique_lock<std::mutex> mlock(mutex_);

	while (queue_.empty())
		cond_.wait(mlock);
	queue_.pop_front();
	queue_size_--;
}

template <typename T> void SharedQueue<T>::push_back(const T& item)
{
	std::unique_lock<std::mutex> mlock(mutex_);

	queue_.push_back(item);
	queue_size_++;
	mlock.unlock();
	cond_.notify_one();
}

template <typename T> void SharedQueue<T>::push_back(T&& item)
{
	std::unique_lock<std::mutex> mlock(mutex_);

	queue_.push_back(std::move(item));
	queue_size_++;
	mlock.unlock();
	cond_.notify_one();

}

template <typename T> int SharedQueue<T>::size()
{
	std::unique_lock<std::mutex> mlock(mutex_);

	int size = queue_.size();
	mlock.unlock();
	return size;
}

template <typename T> unsigned int SharedQueue<T>::counted_size()
{
	return queue_size_;
}

#endif // SHARED_QUEUE_HPP 
