
    # =========================================================================
    # Pseudo Labeling
    # =========================================================================
    def pseudo_label(self):
        """
        Details
        """    
        got_pseudo_label= False
        while not got_pseudo_label:

            # get preds from unlabeled data
            unlabeled_data = next(self._trainer._unlabeled_data_loader_iter)
            student_image = unlabeled_data[0][1]["strong_image"]
            unlabeled_data[0] = unlabeled_data[0][0]

            preds = self.model_teacher(unlabeled_data)
            preds = preds[0]["instances"].to("cpu")

            # process predictions
            pred_scores = preds.scores.detach().numpy()
            pred_masks = preds.pred_masks.detach().numpy()/255
            pred_boxes = preds.pred_boxes.tensor.detach().numpy()

            # fileter masks by pred score
            cf_pred_scores = []
            cf_pred_masks = []
            cf_pred_boxes = []
            
            for i in range(len(pred_scores)):
                if pred_scores[i] < 0.5:
                    continue
                cf_pred_scores.append(pred_scores[i])
                cf_pred_masks.append(pred_masks[i])
                cf_pred_boxes.append(pred_boxes[i])

            # filter masks by metric
            mf_pred_scores = []
            mf_pred_masks = []
            mf_pred_boxes = []
            for i in range(len(cf_pred_scores)):
                # get confidence score and mask
                conf_score = cf_pred_scores[i]
                mask = cf_pred_masks[i]

                # check mask
                binary_mask = np.where(mask >= 0.5, 1, 0)
                area =  np.count_nonzero(binary_mask)

                if area < 50:
                    continue

                # get volumetric symetry metric
                higher_volume = mask[mask >= 0.5] - 0.5
                lower_volume = np.minimum(mask, 0.5)
                vol_sym = conf_score * (higher_volume.sum()/lower_volume.sum())**2  

                #print(vol_sym)
                if vol_sym < self.cfg.PSEUDO_LABELING.METRIC_THRESHOLD:
                    continue
                
                mf_pred_scores.append(cf_pred_scores[i])
                mf_pred_masks.append(binary_mask)
                mf_pred_boxes.append(cf_pred_boxes[i])

            if len(mf_pred_scores) == 0:
                continue
        
            # Data Post Processing
            # get_polygone_masks
            raw_polygons = self.masks_to_polygone_masks(mf_pred_masks)
            all_polygons = []
            for polys in raw_polygons:
                good_polys = []
                for poly in polys:
                    if len(poly) > 4:
                        good_polys.append(poly)
                all_polygons.append(good_polys)
            
            polygone_masks = PolygonMasks(all_polygons)
            mf_pred_boxes = np.array(mf_pred_boxes)
            boxes = Boxes(torch.tensor(mf_pred_boxes).float())
            instances = Instances((unlabeled_data[0]["height"], unlabeled_data[0]["width"]))
            instances.gt_boxes = boxes
            instances.gt_masks = polygone_masks
            instances.gt_classes = torch.tensor([0]*len(polygone_masks))

            got_pseudo_label = True

        # add intances to unlabeled_data
        unlabeled_data[0]["image"] = student_image
        unlabeled_data[0]["instances"] = instances

        # return data
        return unlabeled_data