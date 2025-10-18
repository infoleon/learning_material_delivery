





def run_all(send_thr, receiv_thr, result_queue):
    #print("Starting threads")
    try:
        send_thr.start()
        receiv_thr.start()
        #print("Threads started")
    except Exception as e:
        print(f"Error starting threads: {e}")
        
    try:
        #print("Joining threads")
        send_thr.join()
        receiv_thr.join()
        #print("Threads joined")
    except Exception as e:
        print(f"Error joining threads: {e}")
        
    # Retrieve all items from the queue
    result_list = []
    while not result_queue.empty():
        result_list.append(result_queue.get())
        
    return result_list












