    def run_step(self):
        """
        Detials
        """
        # setup run step: get current iter and start timer
        self._trainer.iter = self.iter
        start = time.perf_counter()

        # always collect labelled data:
        labeled_data = next(self._trainer._labeled_data_loader_iter)
        data_time = time.perf_counter() - start
        
        # if in pre train iteration stage only carry out supervised forward pass
        if self.pre_training and self.iter < self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS:
            loss_dict = self.model(labeled_data)
            losses = sum(loss_dict.values())
        # else do pseudo labeling
        else:
            # if burn in and in burn in range
            if self.burn_in and self.iter < self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS + self.cfg.PSEUDO_LABELING.BURN_IN_ITERS:
                # in first instance load student weights to teacher and initialise burn in weights for student. otherwise do nothing. teache is forzen
                if self.iter == self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS:
                    self._update_teacher_model(keep_rate=0.00)
                    # ADD HERE LOAD STUDENT WEIGHTS
            # elif in distilation range with burn in present
            elif self.burn_in and self.iter > self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS + self.cfg.PSEUDO_LABELING.BURN_IN_ITERS:
                # distilation after burn in so load best student. teacher is already in place, in the first instance, afterward carry out distilation with given ema keep rate:
                if self.iter == self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS + self.cfg.PSEUDO_LABELING.BURN_IN_ITERS:
                    # ADD BEST LOAD STUDENT WEIGHTS
                elif(self.iter - self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS + self.cfg.PSEUDO_LABELING.BURN_IN_ITERS) % self.cfg.PSEUDO_LABELING.PSEUDO_UPDATE_FREQ == 0:
                    self._update_teacher_model(keep_rate = self.cfg.PSEUDO_LABELING.EMA_KEEP_RATE)
            # elif there is no burn in
            elif not self.burn_in:
                # no student burn in so update teacher in first instance else carry out ema weight transfer with given keep rate
                if self.iter == self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS:
                    self._update_teacher_model(keep_rate=0.00)
                elif(self.iter - self.cfg.PSEUDO_LABELING.PRE_TRAIN_ITERS) % self.cfg.PSEUDO_LABELING.PSEUDO_UPDATE_FREQ == 0:
                    self._update_teacher_model(keep_rate = self.cfg.PSEUDO_LABELING.EMA_KEEP_RATE)
        
            # get pseduo labeled data
            pseudo_labeled_data = self.pseudo_label()

            # forward pass of on labeled and unlabeled data
            record_dict = {}
            labeled_loss_dict = self.model(labeled_data)
            unlabeled_loss_dict = self.model(pseudo_labeled_data)
            record_dict["labeled"] = labeled_loss_dict
            record_dict["unlabeled"] = unlabeled_loss_dict

            # process losses
            loss_dict = {}

            for key in record_dict.keys():
                # weighting here later
                loss_dict[key] = sum(record_dict[key].values())

            losses = sum(loss_dict.values())

        if self.cfg.PSEUDO_LABELING.METRIC_USE == "dynamic":
            if self.iter % self.cfg.TEST.EVAL_PERIOD == 0:
                if self.iter != 0:
                    self.metric_thresh = self.metric_mean_val - self.cfg.PSEUDO_LABELING.METRIC_OFFSET
                    print("### NEW_THRESH ######################################")
                    print(self.metric_thresh)
                    self.metric_mean_count = 0
                    self.metric_mean_acc = 0


        self.optimizer.zero_grad()
        losses.backward()
        #self.after_backward()
        self.optimizer.step()