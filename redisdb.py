
import redis as rd


class RedisDB(object):
    
    rc = rd
     
    # ******************* Constructor ************************************************* OK OK
    def __init__(self):
     
        pass
     
    # ******************* open_db Function ********************************************* OK OK
    def open_db(self):
         
        # self.rc = rd.StrictRedis(host='localhost', port=6379, db=0)
        self.rc = rd.StrictRedis(host='192.168.1.152', port=6379, db=0)
        
        pass
     
    # ******************* close_db Function ******************************************** OK OK
    def close_db(self):
         
        pass
     
    # ******************* set_value Function ******************************************* OK OK
    def set_value(self, key, value):
         
        self.rc.set(key, value)
         
        pass

    # ******************* get_value Function ******************************************* OK OK
    def get_value(self, key):

        value = self.rc.get(key)

        return value

        pass
    
    # ******************* set_value_list Function *************************************** OK OK
    def set_value_list(self, key_list, value_list):
        
        for i in range(len(key_list)):
            
            self.rc.set(key_list[i], value_list[i])
         
        pass

    # ******************* get_value_list Function *************************************** OK OK
    def get_value_list(self, key_list):
        
        value_list = []
        
        for key in key_list:
            value_list.append(self.rc.get(key))
         
        return value_list

        pass
    
    # ******************* push_value Function ****************************************** OK OK
    def push_value(self, key, value, work_mode=True):
        
        if work_mode:
            self.rc.rpush(key, value)
        else:
            self.rc.lpush(key, value)
        pass
    
    # ******************* push_value_list Function ********************************** OK OK
    def push_value_list(self, key, value_list, work_mode=True):

        if work_mode:
            for value in value_list:
                self.rc.rpush(key, value)
        else:
            for value in value_list:
                self.rc.lpush(key, value)
        pass
    
    # ******************* pop_value Function ******************************************* OK OK
    def pop_value(self, key, work_mode=True):
         
        if work_mode:
            value = self.rc.lpop(key)
        else:
            value = self.rc.rpop(key)
            
        return value
        pass
    
    # ******************* pop_value_list Function ******************************************* OK OK
    def pop_value_list(self, key, qt, work_mode=True):
        
        value_list = []
                
        if work_mode:
            for i in range(qt):
                value_list.append(self.rc.lpop(key))
        else:
            for i in range(qt):
                value_list.append(self.rc.rpop(key))
            
        return value_list
        pass
    
    # ******************* remove_key Function ****************************************** OK OK
    def remove_key(self, key):
        
        return self.rc.delete(key)
        
        pass
    
    # ******************* get_all_elements Function ************************************* OK OK
    def get_all_elements(self, key):
        
        value_list = self.rc.lrange(key, 0, -1)

        return value_list
        pass
    
    # ******************* list_length Function ***************************************** OK OK
    def list_length(self, key):
        
        return self.rc.llen(key)
        
        pass
