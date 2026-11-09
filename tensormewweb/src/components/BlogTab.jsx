import React from "react";
import { BookOpen } from "lucide-react";

export default function BlogTab() {
  return (
    <div className="blog-view">
      <div className="blog-empty-state-wrap">
        <div className="gsc-dialog-box blog-empty-dialog">
          <div className="dialog-title">TECHNICAL BLOG &amp; DEVLOGS</div>
          <div className="empty-content-inner">
            <BookOpen size={36} className="empty-icon" />
            <h3 className="empty-headline">RESEARCH PAPERS &amp; FIELD LOGS IN PROGRESS</h3>
            <p className="empty-desc">
              Detailed technical write-ups covering TensorMew's 13.2M parameter recurrent spatial policy,
              banked Game Boy Color memory reverse engineering, and Johto Gym leader navigation
              are currently being compiled and will be published here soon.
            </p>
          </div>
          <div className="dialog-prompt-arrow">▼</div>
        </div>
      </div>
    </div>
  );
}
