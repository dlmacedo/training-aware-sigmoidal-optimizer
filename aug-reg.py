            #"""
            # preparing structural augmentation...
            if self.args.loss.split("_")[4] != "no":
                print("augmentation!!!")
                #print("entering 1!!!")
                ####lam = np.random.beta(args.beta, args.beta)
                #batch_slice = 1/self.args.loss.split("_")[4][0]
                #slice_size = inputs.size()[0]//int(self.args.loss.split("_")[4][0])
                #slice_size = 2**int(self.args.loss.split("_")[4][0])
                #slice_size = inputs.size(0)//(2**int(self.args.loss.split("_")[4][0]))
                ###############################
                ###############################
                slice_size = inputs.size(0)//2
                ###############################
                ###############################
                #print("SLICE SIZE:", slice_size)
                W = inputs.size(2)
                H = inputs.size(3)
                rand_index = torch.randperm(slice_size).cuda()
                #print(rand_index)
                ####target_a = target
                ####target_b = target[rand_index]
                ####################################################################################          
                ####################################################################################          
                #plt.imshow(inputs[0].cpu().permute(1, 2, 0))  
                #plt.show()      
                #plt.imshow(inputs[rand_index[0]].cpu().permute(1, 2, 0))  
                #plt.show()      
                ####################################################################################          
                ####################################################################################          
                #"""
                # The bellow appears to work fine with eml...
                #if self.args.loss.startswith("eml"):
                if self.criterion.type.startswith("eml"):
                    #print("#############################")
                    #print(self.criterion.type)
                    #print(self.criterion.weights.size(0))
                    #print("#############################")

                    #bbx1, bby1, bbx2, bby2 = utils.rand_bbox(inputs.size(), 0.85)
                    #inputs[slice_size:, :, bbx1:bbx2, bby1:bby2] = torch.zeros(inputs.size(0)-slice_size, 3, bbx1:bbx2, bby1:bby2) + 0.5  
                    #plt.imshow(inputs[slice_size].cpu().permute(1, 2, 0))  
                    #plt.show()      
                    #plt.imshow(inputs[inputs.size(0)-1].cpu().permute(1, 2, 0))  
                    #plt.show()      

                    #if int(self.criterion.weights.size(0)) < 50:
                    if int(self.criterion.weights.size(1)) < 50:
                        """
                        # only lx!!!
                        r = np.random.rand(1)
                        if r < 0.5:
                            #if (0 <= r < 0.25):
                            print("data1")
                            inputs[:slice_size, :, int(W/2):, :] = inputs[rand_index, :, int(W/2):, :]
                        else:
                            #elif (0.25 <= r < 0.5):
                            print("data2")
                            inputs[:slice_size, :, :, int(H/2):] = inputs[rand_index, :, :, int(H/2):]
                        """
                        #"""
                        r = np.random.rand(1)
                        if (0 <= r < 0.33):
                            print("data1$$$")
                            inputs[:slice_size, :, int(W/2):, :] = inputs[rand_index, :, int(W/2):, :]
                        elif (0.33 <= r < 0.66):
                            print("data2$$$")
                            inputs[:slice_size, :, :, int(H/2):] = inputs[rand_index, :, :, int(H/2):]
                        #elif (0.5 <= r < 0.75):
                        #    print("noise3+++")
                        #    #inputs[:slice_size] = 0.1 * torch.rand(slice_size, 3, int(W), int(H)) # mixed
                        #    inputs[:slice_size] = 0.05 * torch.rand(slice_size, 3, int(W), int(H)) # supermixed
                        #    self.batch_normalize(inputs[:slice_size])
                        else:
                            print("gn0.1x66++++++++++++++++++++++++++++++++++")
                            ##inputs[:slice_size] = torch.rand(slice_size, 3, int(W), int(H))
                            ##inputs[:slice_size] = utils.perlin_noise(slice_size=slice_size)
                            inputs[:slice_size] = (0.1 * torch.randn(slice_size, 3, int(W), int(H))) + 0.5 # gnoise0.1
                            ####rand_base = torch.rand(slice_size, 3).unsqueeze(2).unsqueeze(3)
                            ####inputs[:slice_size] = 0.05 * torch.randn(slice_size, 3, int(W), int(H)) + rand_base # gnoise0.1rand
                            inputs[:slice_size] = torch.clamp(inputs[:slice_size], 0, 1)
                            ##print(inputs[0])
                            ##plt.imshow(inputs[0].cpu().permute(1, 2, 0))
                            ##plt.show()
                            #inputs[:slice_size] = torch.rand(slice_size, 3, int(W), int(H))
                            self.batch_normalize(inputs[:slice_size])
                        #"""
                        """
                        noise = utils.perlin_noise()
                        print(noise.cpu().permute(1, 2, 0).size())
                        print(noise)
                        plt.imshow(noise.cpu().permute(1, 2, 0))
                        plt.show()
                        """
                        """
                        r = np.random.rand(1)
                        # only noise!!! Forget this forever!!!
                        if r < 0.5:
                            #elif (0.5 <= r < 0.75):
                            print("only+noise1")
                            inputs[:slice_size] = 0.1 * torch.rand(slice_size, 3, int(W), int(H))
                            self.batch_normalize(inputs[:slice_size])
                        else:
                            #elif (0.75 <= r <= 1):
                            print("only+noise2")
                            inputs[:slice_size] = 0.1 * torch.randn(slice_size, 3, int(W), int(H)) + 0.5
                            inputs[:slice_size] = torch.clamp(inputs[:slice_size], 0, 1)
                            self.batch_normalize(inputs[:slice_size])
                        """
                    else:
                        """
                        r = np.random.rand(1)
                        if (0 <= r < 0.33):
                            print("data1$$$")
                            inputs[:slice_size, :, int(W/2):, :] = inputs[rand_index, :, int(W/2):, :]
                        elif (0.33 <= r < 0.66):
                            print("data2$$$")
                            inputs[:slice_size, :, :, int(H/2):] = inputs[rand_index, :, :, int(H/2):]
                        #elif (0.5 <= r < 0.75):
                        #    print("noise3+++")
                        #    inputs[:slice_size] = 0.1 * torch.rand(slice_size, 3, int(W), int(H)) # mixed
                        #    #inputs[:slice_size] = 0.05 * torch.rand(slice_size, 3, int(W), int(H)) # supermixed
                        #    self.batch_normalize(inputs[:slice_size])
                        else:
                            print("noise$$$")
                            inputs[:slice_size] = (0.1 * torch.randn(slice_size, 3, int(W), int(H))) + 0.5 # gnoise0.1
                            inputs[:slice_size] = torch.clamp(inputs[:slice_size], 0, 1)
                            self.batch_normalize(inputs[:slice_size])
                        """
                        """
                        # only data!!! Forget this forever!!!
                        r = np.random.rand(1)
                        if r < 0.5:
                            #if (0 <= r < 0.25):
                            print("only+data1")
                            inputs[:slice_size, :, int(W/2):, :] = inputs[rand_index, :, int(W/2):, :]
                        else:
                            #elif (0.25 <= r < 0.5):
                            print("only+data2")
                            inputs[:slice_size, :, :, int(H/2):] = inputs[rand_index, :, :, int(H/2):]
                        """
                        #print("gn0.1_100!!!!!!!!!!!!!!!!!!!")
                        #inputs[:slice_size] = ( 0.1 * torch.randn(slice_size, 3, int(W), int(H))) + 0.5 # gnoise0.1
                        ##rand_base = torch.rand(slice_size, 3).unsqueeze(2).unsqueeze(3)
                        ##inputs[:slice_size] = ( 0.1 * torch.randn(slice_size, 3, int(W), int(H))) + rand_base # gnoise0.1rand
                        #inputs[:slice_size] = torch.clamp(inputs[:slice_size], 0, 1)
                        #self.batch_normalize(inputs[:slice_size])
                        #########################################
                        print("gn0.1x0++++++++++++++++++++++++++++++++++")
                        ##inputs[:slice_size] = torch.rand(slice_size, 3, int(W), int(H))
                        ##inputs[:slice_size] = utils.perlin_noise(slice_size=slice_size)
                        inputs[:slice_size] = (0.1 * torch.randn(slice_size, 3, int(W), int(H))) + 0.5 # gnoise0.1
                        ####rand_base = torch.rand(slice_size, 3).unsqueeze(2).unsqueeze(3)
                        ####inputs[:slice_size] = 0.05 * torch.randn(slice_size, 3, int(W), int(H)) + rand_base # gnoise0.1rand
                        inputs[:slice_size] = torch.clamp(inputs[:slice_size], 0, 1)
                        ##print(inputs[0])
                        ##plt.imshow(inputs[0].cpu().permute(1, 2, 0))
                        ##plt.show()
                        #inputs[:slice_size] = torch.rand(slice_size, 3, int(W), int(H))
                        self.batch_normalize(inputs[:slice_size])
                        ########################################
                        """
                        print("pn_0_x!!!!!!!!!!!!!!!!!!!")
                        #rand_base = torch.rand(slice_size, 3).unsqueeze(2).unsqueeze(3)
                        inputs[:slice_size] = utils.perlin_noise(slice_size=slice_size) #+ rand_base
                        #plt.imshow(inputs[0].cpu().permute(1, 2, 0))
                        #plt.show()
                        self.batch_normalize(inputs[:slice_size])
                        """
                        """
                        # right noise!!!
                        r = np.random.rand(1)
                        if r < 0.5:
                            #elif (0.5 <= r < 0.75):
                            print("noise1!!!!!!!!!!!!")
                            #inputs[:slice_size] = 0.1 * torch.rand(slice_size, 3, int(W), int(H)) # right noise!!!
                            inputs[:slice_size] = 0.1 * torch.rand(slice_size, 3, int(W), int(H)) + 0.45 # new right noise!!! <<<<====
                            self.batch_normalize(inputs[:slice_size])
                        else:
                            #elif (0.75 <= r <= 1):
                            print("noise2!!!!!!!!!!!!")
                            inputs[:slice_size] = 0.1 * torch.randn(slice_size, 3, int(W), int(H)) + 0.5
                            inputs[:slice_size] = torch.clamp(inputs[:slice_size], 0, 1)
                            self.batch_normalize(inputs[:slice_size])
                        """
                #"""
                #"""
                # The bellow appears to work fine with dml...
                if self.args.loss.startswith("dml"):
                    #print("augmentation")
                    ####bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                    #bbx1, bby1, bbx2, bby2 = utils.rand_bbox(inputs.size(), 0.5)
                    ####inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]  
                    #inputs[:slice_size, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]  
                    ####adjust lambda to exactly match pixel ratio!!!
                    ##box_area = ((bbx2 - bbx1) * (bby2 - bby1)) / (inputs.size()[-1] * inputs.size()[-2])
                    ##print(box_area)
                    ##box_area = ((bbx2 - bbx1) * (bby2 - bby1)) / (W * H)
                    ##print(box_area)
                    #"""
                    r = np.random.rand(1)
                    if r < 0.5:
                        #if (0 <= r < 0.25):
                        print("data1")
                        inputs[:slice_size, :, int(W/2):, :] = inputs[rand_index, :, int(W/2):, :]
                    else:
                        #elif (0.25 <= r < 0.5):
                        print("data2")
                        inputs[:slice_size, :, :, int(H/2):] = inputs[rand_index, :, :, int(H/2):]
                    #"""
                    """
                    ####r = np.random.rand(1)
                    #if r < 0.5:
                    elif (0.5 <= r < 0.75):
                        print("3")
                        inputs[:slice_size] = 0.1 * torch.rand(slice_size, 3, int(W), int(H))
                        self.batch_normalize(inputs[:slice_size])
                    else:
                        print("4")
                        inputs[:slice_size] = 0.1 * torch.randn(slice_size, 3, int(W), int(H)) + 0.5
                        inputs[:slice_size] = torch.clamp(inputs[:slice_size], 0, 1)
                        self.batch_normalize(inputs[:slice_size])
                    """
                    """
                    r = np.random.rand(1)
                    if r < 0.5:
                        temp_inputs = inputs[:slice_size, :, int(W/2):, :].clone()
                        inputs[:slice_size, :, int(W/2):, :] = inputs[:slice_size, :, :int(W/2), :]
                        inputs[:slice_size, :, :int(W/2), :] = temp_inputs
                    else:
                        temp_inputs = inputs[:slice_size, :, :, int(W/2):].clone()
                        inputs[:slice_size, :, :, int(W/2):] = inputs[:slice_size, :, :, :int(W/2)]
                        inputs[:slice_size, :, :, :int(W/2)] = temp_inputs
                    """
                #"""
                ####################################################################################          
                ####################################################################################          
                #print(inputs[0].cpu().permute(1, 2, 0))
                """
                plt.imshow(inputs[0].cpu().permute(1, 2, 0))  
                plt.show()      
                plt.imshow(inputs[slice_size].cpu().permute(1, 2, 0))  
                plt.show()      
                """
                ##########################################################################          
                ##########################################################################          
            #"""














            #"""
            # executing structural augmentation...
            if self.args.loss.split("_")[4] != "no":
                print("augmentation!!!")
                #print("entering 3!!!")
                #print(slice_size)
                #augmentation_multiplier = float(self.args.loss.split("_")[4][1:])      
                #augmentation_multiplier = 1/(2**int(self.args.loss.split("_")[4][1:]))                   
                augmentation_multiplier = float(self.args.loss.split("_")[4])                   
                #print(augmentation_multiplier)
                #################################################################
                #################################################################
                #################################################################
                #################################################################
                ####augmentation_outputs = odd_outputs[:slice_size] #### <<<<====
                #################################################################
                #################################################################
                #################################################################
                #################################################################
                uniform_dist = torch.Tensor(slice_size, self.args.number_of_model_classes).fill_((1./self.args.number_of_model_classes)).cuda()
                kl_divergence = F.kl_div(F.log_softmax(odd_outputs[:slice_size], dim=1), uniform_dist, reduction='batchmean')
                augmentation = augmentation_multiplier * kl_divergence
                ############################################################
                #uniform_dist = torch.Tensor(slice_size, self.args.number_of_model_classes).fill_((1./self.args.number_of_model_classes)).cuda()
                #kl_divergence = F.kl_div(torch.log(uniform_dist), F.softmax(odd_outputs[:slice_size], dim=1), reduction='batchmean')
                #augmentation = augmentation_multiplier * kl_divergence
                ############################################################
                ######## cross-entropy from softmax distribution to uniform distribution
                ########loss += 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
                ########cross_entropy = -(odd_outputs[:slice_size].mean(1) - torch.logsumexp(odd_outputs[:slice_size], dim=1)).mean()
                ########augmentation = augmentation_multiplier * cross_entropy
                ############################################################
                #cross_entropy = -(1./self.args.number_of_model_classes * F.log_softmax(odd_outputs[:slice_size], dim=1)).sum(dim=1).mean()
                #augmentation = augmentation_multiplier * cross_entropy
                ############################################################
                ########cross_entropy = -(F.softmax(odd_outputs[:slice_size], dim=1) * math.log(1./self.args.number_of_model_classes)).sum(dim=1).mean()
                ########augmentation = augmentation_multiplier * cross_entropy
                ############################################################
                #entropy = utils.entropies_from_logits(odd_outputs[:slice_size]).mean()
                #augmentation = - augmentation_multiplier * entropy
                ############################################################
                ########entropy = utils.entropies_from_logits(odd_outputs[:slice_size]).mean()
                ########augmentation = augmentation_multiplier * entropy
                ############################################################
                inputs = inputs[slice_size:]
                targets = targets[slice_size:]
                outputs = outputs[slice_size:]
                odd_outputs = odd_outputs[slice_size:]
                loss += augmentation
                #print(augmentation.item())
            else:
                augmentation = torch.tensor(0)
            #"""
