import inspect
import torch

class OptimizerWrapper:
    """
    A wrapper to make optimizer more concise
    """

    def __init__(self, model, optimizer, lr_scheduler=None,clip = False):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.clip = clip



    def step(self, inputs, labels):
        self.zero_grad()
        loss = self.model.loss(inputs, labels)
        # print('optimizer  :'+str(loss))
        if self.clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        loss.backward()
        a = self.optimizer.step()
        torch.cuda.empty_cache() 
        #print('grade   ' + str(self.model.features[0].weight.grad.sum()))
        return a, loss

    def step2(self, inputs, labels, masks):
        self.zero_grad()
        loss = self.model.loss(inputs, labels)
        # print('optimizer  :' + str(loss))
        loss.backward()
        return self.optimizer.step(masks = masks), loss
    
    
    def step3(self, inputs, labels, rate):
        self.zero_grad()
        # print(len(labels))
        loss = self.model.loss(inputs, labels, rate = rate)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        loss.backward()
        a = self.optimizer.step()
        torch.cuda.empty_cache() 
        return  a

    def zero_grad(self):
        self.model.zero_grad()

    def lr_scheduler_step(self, loss=None):#but self.lr_scheduler is None
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr





    def get_last_lr(self):
        if self.lr_scheduler is None:
            return self.optimizer.defaults["lr"]
        else:
            return self.optimizer.param_groups[0]['lr']
