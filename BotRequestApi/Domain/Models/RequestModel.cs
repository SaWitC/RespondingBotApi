using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Domain.Models
{
    public class RequestModel
    {
        [LoadColumn(1), ColumnName("Message")]
        public string Message { get; set; }

        [LoadColumn(2), ColumnName("Response")]
        public string Response { get; set; }
    }
}
